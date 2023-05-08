#if defined(WIN32) || defined(_WIN32) || defined(_WIN32_) || defined(WIN64) || defined(_WIN64) || defined(_WIN64_)
#define DECEXT _declspec(dllexport)
#else
#define DECEXT
#endif

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif
    DECEXT void *go_bert_load_from_file(const char *fname);

    DECEXT void go_bert_free(void *ctx);

    DECEXT int go_bert_n_embd(void *ctx);

    DECEXT void go_bert_encode_batch(void *ctx,
                                     int n_threads,
                                     int n_batch_size,
                                     int n_inputs,
                                     const char **texts,
                                     float **embeddings);

    DECEXT void go_bert_encode(
        void *ctx,
        int n_threads,
        const char *texts,
        float *embeddings);

    DECEXT float *go_bert_eval_batch(
        void *ctx,
        int n_threads,
        int n_batch_size,
        int n_inputs,
        int *batch_tokens_flat,
        int *n_tokens);

#ifdef __cplusplus
}
#endif