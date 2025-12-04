//
// `LongMen` - 'ONNX Model inference in c++'
// Copyright (C) 2019 - present timepi <timepi123@gmail.com>
// LongMen is provided under: GNU Affero General Public License (AGPL3.0)
// https://www.gnu.org/licenses/agpl-3.0.html unless stated otherwise.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//

#ifndef LONGMEN_H_
#define LONGMEN_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file longmen.h
 * @brief C API for LongMen ONNX inference engine
 *
 * Provides a C-compatible interface for deep learning model inference,
 * enabling integration with C programs, FFI bindings (Python ctypes, Rust FFI),
 * and other language ecosystems.
 *
 * @par Thread Safety
 * - longmen_create() is not thread-safe
 * - longmen_release() is not thread-safe
 * - longmen_forward() is thread-safe (can be called concurrently on same model)
 * - longmen_reflush() is thread-safe (atomic pool swap)
 *
 * @par Memory Management
 * - Caller must call longmen_release() to free model resources
 * - Input strings do not need to be null-terminated (length-specified)
 * - Output scores point to internal buffers (valid until next forward call)
 *
 * @par Error Handling
 * - Functions return NULL or -1 on error
 * - Check return values before using results
 * - Errors are logged via glog (check logs for details)
 */

/**
 * @brief Creates a new LongMen model instance
 *
 * Initializes the inference engine by loading configuration files and
 * ONNX model from the specified working directory.
 *
 * @param workdir Path to working directory (not null-terminated)
 * @param len Length of workdir string in bytes
 *
 * @return Opaque model handle on success, NULL on failure
 *
 * @note Caller must call longmen_release() to free resources
 * @note Not thread-safe (do not call concurrently)
 * @note Logs errors via glog on failure
 *
 * @par Required Directory Structure
 * @code
 * workdir/
 * ├── meta.json          # Model configuration
 * ├── features.json      # Feature processing config
 * ├── model.onnx         # ONNX model file
 * └── embeddings/        # Optional embedding tables
 * @endcode
 *
 * @par Example (C)
 * @code
 * const char* workdir = "/path/to/model";
 * void* model = longmen_create((char*)workdir, strlen(workdir));
 * if (model == NULL) {
 *     fprintf(stderr, "Failed to create model\n");
 *     return -1;
 * }
 * // Use model...
 * longmen_release(model);
 * @endcode
 *
 * @par Example (Python ctypes)
 * @code{.py}
 * import ctypes
 * lib = ctypes.CDLL("liblongmen.so")
 * lib.longmen_create.argtypes = [ctypes.c_char_p, ctypes.c_int32]
 * lib.longmen_create.restype = ctypes.c_void_p
 *
 * workdir = b"/path/to/model"
 * model = lib.longmen_create(workdir, len(workdir))
 * if not model:
 *     raise RuntimeError("Failed to create model")
 * @endcode
 */
void *longmen_create(char *workdir, int32_t len);

/**
 * @brief Releases model resources
 *
 * Destroys the model instance and frees all associated resources including
 * ONNX Runtime session, feature pools, and memory arena.
 *
 * @param model Model handle returned by longmen_create() (can be NULL)
 *
 * @note Safe to call with NULL pointer (no-op)
 * @note Not thread-safe (do not call concurrently with other operations)
 * @note After calling, model handle is invalid and must not be used
 *
 * @par Example (C)
 * @code
 * void* model = longmen_create(workdir, len);
 * // Use model...
 * longmen_release(model);
 * model = NULL;  // Good practice
 * @endcode
 */
void longmen_release(void *model);

/**
 * @brief Hot-swaps the feature pool
 *
 * Atomically replaces the current feature pool with a new one loaded from
 * the specified file. Ongoing inference operations continue using the old
 * pool until completion.
 *
 * @param model Model handle (must not be NULL)
 * @param path Path to new feature file (not null-terminated)
 * @param len Length of path string in bytes
 * @param version Version number for the new pool
 *
 * @note Thread-safe (atomic pool swap)
 * @note Safe to call with NULL model (no-op)
 * @note Old pool is kept alive until all references are released
 * @note Does not return error status (check logs for errors)
 *
 * @par File Format
 * Each line: `item_id\t{"feature1": value1, ...}`
 *
 * @par Example (C)
 * @code
 * const char* pool_path = "/path/to/items.txt";
 * longmen_reflush(model, (char*)pool_path, strlen(pool_path), 1);
 * // New inference will use new pool
 * @endcode
 *
 * @par Example (Python ctypes)
 * @code{.py}
 * pool_path = b"/path/to/items.txt"
 * lib.longmen_reflush(model, pool_path, len(pool_path), 1)
 * @endcode
 */
void longmen_reflush(void *model, char *path, int32_t len, int64_t version);

/**
 * @brief Performs batch inference
 *
 * Processes user features, looks up item features, generates cross features,
 * and executes model inference to produce scores.
 *
 * @param model Model handle (must not be NULL)
 * @param user_features JSON-encoded user features (not null-terminated)
 * @param len Length of user_features in bytes
 * @param items Array of item ID pointers (char**, each not null-terminated)
 * @param lens Array of item ID lengths (size_t*, one per item)
 * @param size Batch size (number of items)
 * @param[out] scores Array of float pointers to receive scores (float**)
 * @param[out] version Pool version used for this inference
 *
 * @return 0 on success, -1 on failure
 *
 * @note Thread-safe (can be called concurrently on same model)
 * @note scores[i] will point to internal buffers (valid until next call)
 * @note Missing items are skipped (not an error)
 * @note Sets *version to -1 if no pool is loaded
 *
 * @warning All pointer parameters must be valid (not NULL)
 * @warning size must match the length of items/lens/scores arrays
 * @warning Caller must ensure items[i] points to valid memory of lens[i] bytes
 * @warning Do not free scores[i] pointers (managed internally)
 *
 * @par Example (C)
 * @code
 * // Prepare inputs
 * char user[] = "{\"user_id\": 123, \"age\": 25}";
 * char* items[] = {"item_1", "item_2", "item_3"};
 * size_t lens[] = {6, 6, 6};
 * float* scores[3];
 * int64_t version;
 *
 * // Run inference
 * int ret = longmen_forward(
 *     model,
 *     user, strlen(user),
 *     items, lens, 3,
 *     scores, &version
 * );
 *
 * if (ret == 0) {
 *     for (int i = 0; i < 3; i++) {
 *         printf("Item %d score: %f\n", i, *scores[i]);
 *     }
 *     printf("Pool version: %ld\n", version);
 * } else {
 *     fprintf(stderr, "Inference failed\n");
 * }
 * @endcode
 *
 * @par Example (Python ctypes)
 * @code{.py}
 * # Setup function signature
 * lib.longmen_forward.argtypes = [
 *     ctypes.c_void_p,           # model
 *     ctypes.c_char_p,           # user_features
 *     ctypes.c_int32,            # len
 *     ctypes.POINTER(ctypes.c_char_p),  # items
 *     ctypes.POINTER(ctypes.c_size_t),  # lens
 *     ctypes.c_int32,            # size
 *     ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # scores
 *     ctypes.POINTER(ctypes.c_int64)  # version
 * ]
 * lib.longmen_forward.restype = ctypes.c_int32
 *
 * # Prepare inputs
 * user = b'{"user_id": 123, "age": 25}'
 * items = [b"item_1", b"item_2", b"item_3"]
 * items_arr = (ctypes.c_char_p * len(items))(*items)
 * lens_arr = (ctypes.c_size_t * len(items))(*[len(x) for x in items])
 * scores_arr = (ctypes.POINTER(ctypes.c_float) * len(items))()
 * version = ctypes.c_int64()
 *
 * # Run inference
 * ret = lib.longmen_forward(
 *     model, user, len(user),
 *     items_arr, lens_arr, len(items),
 *     scores_arr, ctypes.byref(version)
 * )
 *
 * if ret == 0:
 *     for i in range(len(items)):
 *         print(f"Item {i} score: {scores_arr[i][0]}")
 *     print(f"Pool version: {version.value}")
 * @endcode
 */
int32_t longmen_forward(void *model, char *user_features, int32_t len,
                        void *items, void *lens, int32_t size, void *scores,
                        int64_t *version);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif // LONGMEN_H_
