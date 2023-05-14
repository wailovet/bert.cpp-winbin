package gobert

import (
	"math/rand"
	"time"

	"github.com/wailovet/easycgo"
	"github.com/wailovet/gotokenizer-bin/gotokenizer"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

type Bret struct {
	ctx       easycgo.ValueInf
	tokenizer *gotokenizer.Tokenizer
	nEmbd     int
}

func NewBret(mFile string) *Bret {
	ctx := BertLoadFromFile(mFile)
	nEmbd := BertNEmbd(ctx)
	return &Bret{ctx: ctx, nEmbd: nEmbd}
}

func (b *Bret) SetTokenizer(tokenizer *gotokenizer.Tokenizer) {
	b.tokenizer = tokenizer
}

func (b *Bret) Encode(text string, nThreads int) []float32 {
	idsU32 := b.tokenizer.EncodeIds(text, true)
	ids := []int{}
	for i := 0; i < len(idsU32); i++ {
		ids = append(ids, int(idsU32[i]))
	}
	ret := BertEvalBatch(b.ctx, nThreads, 1, [][]int{
		ids,
	}, b.nEmbd)
	return ret[0]
}

func (b *Bret) EncodeBatch(texts []string, nThreads int, nBatch int) [][]float32 {
	ids := [][]int{}
	for _, text := range texts {
		idsU32 := b.tokenizer.EncodeIds(text, true)
		ids2 := []int{}
		for i := 0; i < len(idsU32); i++ {
			ids2 = append(ids2, int(idsU32[i]))
		}
		ids = append(ids, ids2)
	}
	ret := BertEvalBatch(b.ctx, nThreads, nBatch, ids, b.nEmbd)
	return ret
}

func (b *Bret) EmbedSize() int {
	return b.nEmbd
}
