package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"
	
	"github.com/schollz/progressbar/v3"
	"github.com/eigengrau01/bladegopher/internal/mlp"
)

func main() {
	net := mlp.CreateNetwork(784, 200, 10, 0.1)

	train := flag.Bool("train", false, "Train the Neural Network")
	file := flag.String("file", "", "File name of 28 x 28 PNG file to evaluate")
	flag.Parse()

	if *train {
		mnistTrain(&net)
		mlp.Save(net)
	}

	if *file != "" {
		printImage(getImage(*file))
		mlp.Load(&net)
		fmt.Println("prediction:", mlp.PredictFromImage(net, *file))
	}
}

func mnistTrain(net *mlp.Network) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()
	bar := progressbar.Default(5)

	for epochs := 0; epochs < 5; epochs++ {
		testFile, _ := os.Open("mnist_dataset/mnist_train.csv")
		r := csv.NewReader(bufio.NewReader(testFile))
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}

			inputs := make([]float64, net.Inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = (x / 255.0 * 0.999) + 0.001
			}

			targets := make([]float64, 10)
			for i := range targets {
				targets[i] = 0.001
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.999

			net.Train(inputs, targets)
		}
		bar.Add(1)
		testFile.Close()
	}
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train: %s\n", elapsed)
}


func printImage(img image.Image) {
	var buf bytes.Buffer
	png.Encode(&buf, img)
	imgBase64Str := base64.StdEncoding.EncodeToString(buf.Bytes())
	fmt.Printf("\x1b]1337;File=inline=1:%s\a\n", imgBase64Str)
}

func getImage(filePath string) image.Image {
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	img, _, err := image.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}
	return img
}
