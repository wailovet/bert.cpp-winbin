#include "binding.h"

#include "bert.h"
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <Windows.h>
#include <algorithm>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <signal.h>
#endif

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
void sigint_handler(int signo)
{
    if (signo == SIGINT)
    {
        _exit(130);
    }
}
#endif

void *go_bert_load_from_file(const char *fname)
{
    return bert_load_from_file(fname);
}

void go_bert_free(void *ctx)
{
    bert_free((bert_ctx *)ctx);
}

int go_bert_n_embd(void *ctx)
{
    return bert_n_embd((bert_ctx *)ctx);
}

void go_bert_encode_batch(void *ctx,
                          int n_threads,
                          int n_batch_size,
                          int n_inputs,
                          const char **texts,
                          float **embeddings)
{
    bert_encode_batch((bert_ctx *)ctx, n_threads, n_batch_size, n_inputs, texts, embeddings);
}

void go_bert_encode(
    void *ctx,
    int n_threads,
    const char *texts,
    float *embeddings)
{
    bert_encode((bert_ctx *)ctx, n_threads, texts, embeddings);
}

float *go_bert_eval_batch(
    void *ctx,
    int n_threads,
    int n_batch_size,
    int n_inputs,
    int *batch_tokens_flat,
    int *n_tokens)
{

    // std::cout << "go_bert_eval_batch..."
    //           << "  n_batch_size:" << n_batch_size << "  n_inputs:" << n_inputs << std::endl;

    int **batch_tokens = new int *[n_inputs];
    float **embeddings = new float *[n_inputs];

    int embedding_size = bert_n_embd((bert_ctx *)ctx);

    int index = 0;

    for (int i = 0; i < n_inputs; i++)
    {
        // std::cout << "n_tokens[" << i << "]:" << n_tokens[i] << std::endl;

        batch_tokens[i] = new int[n_tokens[i]];
        for (int j = 0; j < n_tokens[i]; j++)
        {
            // std::cout << "batch_tokens[" << i << "][" << j << "]:" << batch_tokens_flat[index] << std::endl;
            batch_tokens[i][j] = batch_tokens_flat[index];
            index++;
        }

        embeddings[i] = new float[embedding_size];
    }

    // TODO: Disable batching for now
    n_batch_size = 1;
    /*
    if (n_batch_size > n_inputs) {
        n_batch_size = n_inputs;
    }
    if (n_batch_size > ctx->max_batch_n) {
        bert_resize_ctx(ctx, n_batch_size);
        n_batch_size = ctx->max_batch_n;
    }
    */

    int32_t N = bert_n_max_tokens((bert_ctx *)ctx);

    if (n_batch_size == n_inputs)
    {
        bert_eval_batch((bert_ctx *)ctx, n_threads, n_batch_size, batch_tokens, n_tokens, embeddings);
    }
    else
    {
        // sort the inputs by tokenized length, batch and eval

        std::vector<int> indices;
        indices.reserve(n_inputs);
        for (int i = 0; i < n_inputs; i++)
        {
            indices.push_back(i);
        }

        std::vector<int32_t> sorted_n_tokens = std::vector<int32_t>(n_inputs);

        std::vector<bert_vocab_id *> sorted_tokens(n_inputs);

        std::sort(indices.begin(), indices.end(), [&](int a, int b)
                  { return n_tokens[a] < n_tokens[b]; });

        std::vector<float *> sorted_embeddings(n_inputs);
        memcpy(sorted_embeddings.data(), embeddings, n_inputs * sizeof(float *));

        for (int i = 0; i < n_inputs; i++)
        {
            sorted_embeddings[i] = embeddings[indices[i]];
            sorted_tokens[i] = batch_tokens[indices[i]];
            sorted_n_tokens[i] = n_tokens[indices[i]];
        }

        for (int i = 0; i < n_inputs; i += n_batch_size)
        {
            if (i + n_batch_size > n_inputs)
            {
                n_batch_size = n_inputs - i;
            }
            bert_eval_batch((bert_ctx *)ctx, n_threads, n_batch_size, &sorted_tokens[i], &sorted_n_tokens[i], &sorted_embeddings[i]);
        }
    }

    
    float *embeddings_return = new float[n_inputs * embedding_size];

    index = 0;

    for (int i = 0; i < n_inputs; i++)
    {
        for (int j = 0; j < embedding_size; j++)
        {
            embeddings_return[index] = embeddings[i][j];
            index++;
        }
    }

    for (int i = 0; i < n_inputs; i++)
    {
        delete[] batch_tokens[i];
        delete[] embeddings[i];
    }

    delete[] batch_tokens;
    delete[] embeddings;
    

    return embeddings_return;
}