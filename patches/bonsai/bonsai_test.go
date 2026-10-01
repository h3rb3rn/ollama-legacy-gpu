package gguf

import (
	"bytes"
	"io"
	"testing"
)

// Read real on-disk type IDs, rather than constructing only Go enum values.
// This covers the old bug where private types had zero size and tensor data
// boundaries were calculated incorrectly during an Ollama import.
func TestBonsaiGGUFImport(t *testing.T) {
	for _, tt := range []struct {
		name       string
		kind       uint32
		fileType   uint32
		blockBytes int
	}{
		{"PQ2_0", 142, 141, 34},
		{"PQ2_0", 142, 142, 34},
		{"PTQ1_0", 143, 143, 28},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var b bytes.Buffer
			writeInternalRaw(t, &b, []byte("GGUF"))
			writeInternalRaw(t, &b, uint32(3))
			writeInternalRaw(t, &b, uint64(1))
			writeInternalRaw(t, &b, uint64(2))
			writeInternalString(t, &b, "general.file_type")
			writeInternalRaw(t, &b, typeUint32)
			writeInternalRaw(t, &b, tt.fileType)
			writeInternalString(t, &b, "prism.hadamard.block_size")
			writeInternalRaw(t, &b, typeUint32)
			writeInternalRaw(t, &b, uint32(1024))
			writeInternalString(t, &b, "output.weight")
			writeInternalRaw(t, &b, uint32(2))
			writeInternalRaw(t, &b, uint64(128))
			writeInternalRaw(t, &b, uint64(2))
			writeInternalRaw(t, &b, tt.kind)
			writeInternalRaw(t, &b, uint64(0))
			for b.Len()%32 != 0 {
				b.WriteByte(0)
			}
			payload := bytes.Repeat([]byte{0x55}, 2*tt.blockBytes)
			b.Write(payload)
			path := writeTempFile(t, b.Bytes())
			f, err := Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			info, reader, err := f.TensorReader("output.weight")
			if err != nil {
				t.Fatal(err)
			}
			if info.NumBytes() != int64(len(payload)) {
				t.Fatalf("size = %d", info.NumBytes())
			}
			got, err := io.ReadAll(reader)
			if err != nil || !bytes.Equal(got, payload) {
				t.Fatalf("payload mismatch: %v", err)
			}
			metadata, err := ReadFileMetadata(path, -1)
			if err != nil {
				t.Fatal(err)
			}
			if n, ok := metadata.ExactKeyValue("prism.hadamard.block_size").UintOK(); !ok || n != 1024 {
				t.Fatalf("rotation metadata lost: %d, %v", n, ok)
			}
			if FileType(tt.fileType).String() != tt.name {
				t.Fatal("incorrect file type name")
			}
		})
	}
}

func TestBonsaiRejectsIncompleteQuantizationBlock(t *testing.T) {
	for _, kind := range []TensorType{142, 143} {
		if (TensorInfo{Name: "invalid", Type: kind, Shape: []uint64{127, 2}}).Valid() {
			t.Fatal("accepted a row that cannot contain complete Bonsai quantization blocks")
		}
	}
}
