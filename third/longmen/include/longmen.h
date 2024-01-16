//
// `LongMen` - 'Torch Model inference in c++'
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

#ifndef LONGMAN_H
#define LONGMAN_H

#ifdef __cplusplus
extern "C" {
#endif

void* longmen_new_pool_rows(const char* pool_file,const char* lua_file, const char* luban_file, void * model_ptr);
void longmen_delete_pool_rows(void* ptr);
void *new_longmen_torch_model(const char* model_file, const char* model_meta);
void delete_longmen_torch_model(void *ptr);
void longmen_user_rows_embedding_preforward(void *model_ptr, void *user_rows_ptr, void* pool_rows_ptr);
void longmen_torch_model_inference(void*model_ptr,void* user_rows_ptr, void* pool_rows_ptr, char *items_ptr,
                     void *lens_ptr, int size, float *scores);

#ifdef __cplusplus
} /* end extern "C"*/
#endif

#endif // LONGMAN_H