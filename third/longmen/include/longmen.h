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

void *longmen_new_model(char *path, int plen, char *key, int klen,
                        char *toolkit, int tlen, char *model, int mlen);
void longmen_del_model(void *model);
void longmen_forward(void *model, char *user_features, int len, char *items,
                     void *lens, int size, float *scores);
#ifdef __cplusplus
} /* end extern "C"*/
#endif

#endif // LONGMAN_H