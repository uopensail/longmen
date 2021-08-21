#include "longmen.h"
#include <string>
#include <iostream>

#define BASE64_PAD '='
#define BASE64DE_FIRST '+'
#define BASE64DE_LAST 'z'

/* BASE 64 encode table */
static const char base64en[] = {
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
        'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
        'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
        'w', 'x', 'y', 'z', '0', '1', '2', '3',
        '4', '5', '6', '7', '8', '9', '+', '/',
};

/* ASCII order for BASE 64 decode, 255 in unused character */
static const unsigned char base64de[] = {
        /* nul, soh, stx, etx, eot, enq, ack, bel, */
        255, 255, 255, 255, 255, 255, 255, 255,

        /*  bs,  ht,  nl,  vt,  np,  cr,  so,  si, */
        255, 255, 255, 255, 255, 255, 255, 255,

        /* dle, dc1, dc2, dc3, dc4, nak, syn, etb, */
        255, 255, 255, 255, 255, 255, 255, 255,

        /* can,  em, sub, esc,  fs,  gs,  rs,  us, */
        255, 255, 255, 255, 255, 255, 255, 255,

        /*  sp, '!', '"', '#', '$', '%', '&', ''', */
        255, 255, 255, 255, 255, 255, 255, 255,

        /* '(', ')', '*', '+', ',', '-', '.', '/', */
        255, 255, 255, 62, 255, 255, 255, 63,

        /* '0', '1', '2', '3', '4', '5', '6', '7', */
        52, 53, 54, 55, 56, 57, 58, 59,

        /* '8', '9', ':', ';', '<', '=', '>', '?', */
        60, 61, 255, 255, 255, 255, 255, 255,

        /* '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', */
        255, 0, 1, 2, 3, 4, 5, 6,

        /* 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', */
        7, 8, 9, 10, 11, 12, 13, 14,

        /* 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', */
        15, 16, 17, 18, 19, 20, 21, 22,

        /* 'X', 'Y', 'Z', '[', '\', ']', '^', '_', */
        23, 24, 25, 255, 255, 255, 255, 255,

        /* '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', */
        255, 26, 27, 28, 29, 30, 31, 32,

        /* 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', */
        33, 34, 35, 36, 37, 38, 39, 40,

        /* 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', */
        41, 42, 43, 44, 45, 46, 47, 48,

        /* 'x', 'y', 'z', '{', '|', '}', '~', del, */
        49, 50, 51, 255, 255, 255, 255, 255
};

unsigned int base64_decode(const char *in, unsigned int inlen, unsigned char *out) {
    unsigned int i;
    unsigned int j;
    unsigned char c;

    if (inlen & 0x3) {
        return 0;
    }

    for (i = j = 0; i < inlen; i++) {
        if (in[i] == BASE64_PAD) {
            break;
        }
        if (in[i] < BASE64DE_FIRST || in[i] > BASE64DE_LAST) {
            return 0;
        }

        c = base64de[(unsigned char) in[i]];
        if (c == 255) {
            return 0;
        }

        switch (i & 0x3) {
            case 0:
                out[j] = (c << 2) & 0xFF;
                break;
            case 1:
                out[j++] |= (c >> 4) & 0x3;
                out[j] = (c & 0xF) << 4;
                break;
            case 2:
                out[j++] |= (c >> 2) & 0xF;
                out[j] = (c & 0x3) << 6;
                break;
            case 3:
                out[j++] |= c;
                break;
        }
    }

    return j;
}

int main() {
    auto model = lm_create_model("../test/config.toml", nullptr);
    const char *user_features_bytes_str = "CqEJCgt1X2RfaW5zdGFsbBKRCQqOCQoTY29tLmFuZHJvaWQudmVuZGluZwoTY29tLmZhY2Vib29rLmthdGFuYQoWY29tLnRyYW5zc2lvbi5ub3RlYm9vawoaY29tLmNhbWVyYXNpZGVhcy5pbnN0YXNob3QKFGNvbS5zbmFwY2hhdC5hbmRyb2lkChBjb20ua3dhaS5idWxsZG9nChVjb20uYW5kcm9pZC5kZXNrY2xvY2sKDGNvbS53aGF0c2FwcAoNbmV0LmJhdC5zdG9yZQoUY29tLmFuZHJvaWQuY29udGFjdHMKJGNvbS5uZXdfZmFzdC52cG5fZnJlZS52cG4uc2VjdXJlX3ZwbgoYY29tLnNlbnNldGltZS5mYWNldW5sb2NrChBjb20ucmxrLndlYXRoZXJzChNjb20udGFscGEuaGlicm93c2VyChljb20udHJhbnNzaW9uLnBob25lbWFzdGVyChNjb20udHJhbnNzbmV0LnN0b3JlChZjb20udHJhbnNzaW9uLmNhcmxjYXJlChhjb20uemhpbGlhb2FwcC5tdXNpY2FsbHkKIWNvbS5nb29nbGUuYW5kcm9pZC5hcHBzLm1lc3NhZ2luZwoSY29tLmFuZHJvaWQuZGlhbGVyCiBjb20uZ29vZ2xlLmFuZHJvaWQuYXBwcy5waG90b3NnbwoVY29tLmFmbW9iaS5ib29tcGxheWVyChJjb20uaW5maW5peC54c2hhcmUKJWNvbS5nb29nbGUuYW5kcm9pZC5hcHBzLnlvdXR1YmUubXVzaWMKH2NvbS5nb29nbGUuYW5kcm9pZC5hcHBzLnRhY2h5b24KEWNvbS56YXoudHJhbnNsYXRlChxjb20uZ29vZ2xlLmFuZHJvaWQuYXBwcy5tYXBzCiJjb20uZ29vZ2xlLmFuZHJvaWQuYXBwcy5zZWFyY2hsaXRlChtjb20uZ29vZ2xlLmFuZHJvaWQuY2FsZW5kYXIKGWNvbS5hbmRyb2lkLnNvdW5kcmVjb3JkZXIKF2NvbS50cmFuc3Npb24udGVjbm9zcG90ChNjb20ubWVkaWF0ZWsuY2FtZXJhChpjb20uZ29vZ2xlLmFuZHJvaWQueW91dHViZQoUY29tLmFuZHJvaWQuc2V0dGluZ3MKE2NvbS5hbmRyb2lkLmZtcmFkaW8KCmlvLmZhY2VhcHAKHGNvbS5nb29nbGUuYW5kcm9pZC5hcHBzLmRvY3MKIWNvbS5nb29nbGUuYW5kcm9pZC5hcHBzLmFzc2lzdGFudAoaY29tLnRyYW5zc2lvbi5maWxlbWFuYWdlcngKF2NvbS5hbmRyb2lkLmRvY3VtZW50c3VpChVjb20uZ29vZ2xlLmFuZHJvaWQuZ20KD2NvbS5hbmRyb2lkLnN0awoNY29tLmdhbGxlcnkyMAocY29tLm92aWxleC5jb2FjaGJ1c3NpbXVsYXRvcgoXY29tLmFuZHJvaWQuY2FsY3VsYXRvcjIKEmNvbS5hbmRyb2lkLmNocm9tZQohY29tLmdvb2dsZS5hbmRyb2lkLmFwcHMubmJ1LmZpbGVzCjIKBnVfc19pZBIoCiYKJDAwMDkzZGM5LTA4YzEtNDBhYS05NTBhLWYxMjhmMjU0ZWU4Ng==";
    unsigned char *user_features_ptr = (unsigned char *) calloc(1, strlen(user_features_bytes_str));

    unsigned int out_len = base64_decode((const char *) (user_features_bytes_str),
                                         (unsigned int) (strlen(user_features_bytes_str)),
                                         user_features_ptr);

    auto user_features = lm_create_features((char *) user_features_ptr, out_len);
    const char *item_data =
            "s_ap20153752a6694e948c7f81221f5f7643\0";

    float result = 0.0;
    lm_predict(model, (void *) user_features, (void *) item_data, 1, &result);
    std::cout << result << std::endl;
    //释放features
    lm_release_features(user_features);


    lm_release_model(model);
    return 0;
}

