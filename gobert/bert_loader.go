package gobert

import (
	_ "embed"
	"os"
	"path/filepath"

	"github.com/wailovet/easycgo"
)

var sharedLibrary *easycgo.EasyCgo
var go_bert_load_from_file *easycgo.EasyCgoProc
var go_bert_free *easycgo.EasyCgoProc
var go_bert_n_embd *easycgo.EasyCgoProc
var go_bert_encode_batch *easycgo.EasyCgoProc
var go_bert_encode *easycgo.EasyCgoProc
var go_bert_eval_batch *easycgo.EasyCgoProc

//go:embed bert.cpp.dll
var dllFileAVX2 []byte

func Install() {
	tmpDir := os.TempDir()
	os.WriteFile(filepath.Join(tmpDir, "bert.cpp.dll"), dllFileAVX2, 0666)
	LoadDll(filepath.Join(tmpDir, "bert.cpp.dll"))
}

func LoadDll(dllFile string) {
	sharedLibrary = easycgo.MustLoad(dllFile)
	go_bert_load_from_file = sharedLibrary.MustFind("go_bert_load_from_file")
	go_bert_free = sharedLibrary.MustFind("go_bert_free")
	go_bert_n_embd = sharedLibrary.MustFind("go_bert_n_embd")
	go_bert_encode_batch = sharedLibrary.MustFind("go_bert_encode_batch")
	go_bert_encode = sharedLibrary.MustFind("go_bert_encode")
	go_bert_eval_batch = sharedLibrary.MustFind("go_bert_eval_batch")
}

func BertLoadFromFile(fname string) easycgo.ValueInf {
	return go_bert_load_from_file.Call(fname)
}

func BertFree(ctx easycgo.ValueInf) {
	go_bert_free.Call(ctx)
}

func BertNEmbd(ctx easycgo.ValueInf) int {
	return go_bert_n_embd.Call(ctx).ToInt()
}

func BertEncodeBatch(ctx easycgo.ValueInf, n_threads int, n_batch_size int, texts []string, embeddings [][]float32) {
	go_bert_encode_batch.Call(ctx, n_threads, n_batch_size, len(texts), texts, embeddings)
}

func BertEncode(ctx easycgo.ValueInf, n_threads int, texts string, embeddings []float32) {
	go_bert_encode.Call(ctx, n_threads, texts, embeddings)
}

// DECEXT void go_bert_eval_batch(
//
//	void *ctx,
//	int n_threads,
//	int n_batch_size,
//	int n_inputs,
//	int **batch_tokens,
//	int *n_tokens, );
func BertEvalBatch(ctx easycgo.ValueInf, n_threads int, n_batch_size int, batch_tokens [][]int, embeddings_size int) [][]float32 {
	n_tokens := make([]int32, len(batch_tokens))
	for i, v := range batch_tokens {
		n_tokens[i] = int32(len(v))
	}

	batch_tokens_flat := make([]int32, 0)

	for _, v := range batch_tokens {
		for _, v2 := range v {
			batch_tokens_flat = append(batch_tokens_flat, int32(v2))
		}
	}
	embeddings := go_bert_eval_batch.Call(ctx, n_threads, n_batch_size, len(batch_tokens), batch_tokens_flat, n_tokens)

	retF32s := embeddings.ToFloat32Slice(embeddings_size * len(batch_tokens))

	ret := make([][]float32, len(batch_tokens))
	for i := 0; i < len(batch_tokens); i++ {
		ret[i] = retF32s[i*embeddings_size : (i+1)*embeddings_size]
	}
	return ret
}
