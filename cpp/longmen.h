//
// `LongMen` - 'Torch Model inference in c++'
// Copyright (C) 2019 - present timepi <timepi123@gmail.com>
// LuBan is provided under: GNU Affero General Public License (AGPL3.0)
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

#ifndef LONGMAN_C_H
#define LONGMAN_C_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief create sparse model object
 *
 * @param config_path feature operator config path
 * @param data_path item pool path
 * @param model_path torch model path
 * @param key id field name of item
 * @return void* pointer to sparse model
 */
void *longmen_new(char *config_path, char *data_path, char *model_path,
                  char *key);

void longmen_release(void *ptr);

/**
 * @brief get the scores for sort
 *
 * @param model model ptr
 * @param user_features user feature in bytes
 * @param len user features bytes length
 * @param items items list
 * @param size item num
 * @param scores score results
 */
char *longmen_forward(void *model, char *user_features, int len, void *items,
                      int size, float *scores);

/**
 * @brief relaoad pool
 *
 * @param model model ptr
 * @param data_path item pool path
 */
void longmen_reload(void *model, char *data_path);

#ifdef __cplusplus
} /* end extern "C"*/
#endif

#endif  // LONGMAN_C_H