package llm

import (
	"reflect"
	"strconv"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestLegacyBatchCap(t *testing.T) {
	for _, tt := range []struct {
		name, limit     string
		requested, want int
	}{
		{"disabled", "", 512, 512},
		{"invalid", "invalid", 512, 512},
		{"nonpositive", "0", 512, 512},
		{"cap", "64", 512, 64},
		{"preserve-smaller-request", "64", 32, 32},
		{"cap-upstream-default", "64", 0, 64},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_MAX_BATCH_SIZE", tt.limit)
			opts := api.DefaultOptions()
			opts.NumBatch = tt.requested
			got := appendBatchArgs(nil, opts, false, 1)
			want := []string{"-b", strconv.Itoa(tt.want), "-ub", strconv.Itoa(tt.want)}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("batch arguments = %v, want %v", got, want)
			}
		})
	}
}
