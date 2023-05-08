package main

import (
	"encoding/json"

	"github.com/wailovet/bert.cpp-winbin/gobert"
	"github.com/wailovet/gotokenizer-bin/gotokenizer"
)

func main() {
	gobert.Install()
	gotokenizer.Install()
	bert := gobert.NewBret(`C:\Users\admin\go\src\github.com\wailovet\bert.cpp-winbin\models\ggml-model-q4_0.bin`)
	tokenizer := gotokenizer.NewTokenizerFromFile("tokenizer.json")
	bert.SetTokenizer(tokenizer)
	var s = bert.EncodeBatch(
		[]string{
			"测试一下",
			"名侦探柯南",
		},
		2,
		2,
	)

	s1, _ := json.Marshal(s[0])
	s2, _ := json.Marshal(s[1])

	var b = bert.Encode("我是中国人", 1)

	b1, _ := json.Marshal(b)

	if string(s1) != string(b1) {
		panic("error")
	}

	b = bert.Encode("测试一下", 1)

	b1, _ = json.Marshal(b)

	if string(s2) != string(b1) {
		panic("error")
	}

}
