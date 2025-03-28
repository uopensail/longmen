#ifndef LONGMEN_H_
#define LONGMEN_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Handle type for Longmen model instance
 */
typedef void *LongmenModel;

/**
 * @brief Handle type for data pool
 */
typedef void *LongmenPool;

/**
 * @brief Handle type for inference outputs
 */
typedef void *LongmenOutputs;

/**
 * @brief Create a new model instance
 * @param workdir Path to model workspace directory
 * @return Model handle or NULL on failure
 */
LongmenModel longmen_create_model(const char *workdir);

/**
 * @brief Release model resources
 * @param model Model handle to release
 */
void longmen_release_model(LongmenModel model);

/**
 * @brief Perform batch inference
 * @param model Model handle
 * @param batch_size Number of items per batch
 * @param pool Data pool handle
 * @param user_features User feature string
 * @param items Array of item IDs to process
 * @return Inference outputs handle
 */
LongmenOutputs longmen_serve(LongmenModel model, int batch_size,
                             LongmenPool pool, const char *user_features,
                             const char **items);

/**
 * @brief Create a data pool from dataset
 * @param model Model handle
 * @param data_path Path to material data file
 * @return Data pool handle or NULL on failure
 */
LongmenPool longmen_create_pool(LongmenModel model, const char *data_path);

/**
 * @brief Release data pool resources
 * @param pool Data pool handle to release
 */
void longmen_release_pool(LongmenPool pool);

/**
 * @brief Release inference outputs
 * @param outputs Outputs handle to release
 */
void longmen_release_outputs(LongmenOutputs outputs);

#ifdef __cplusplus
} /* end extern "C"*/
#endif

#endif // LONGMEN_H_
