using System;
using System.IO;

namespace Collect
{
    public static class ProcessFunctions
    {
        public static float[] Add_Data(float[] sample, int Size, float[] x, int Dim)
        {
            float[] temp = new float[Size * Dim];
            
            // Mevcut verileri kopyala
            if (Size > 1)
            {
                Array.Copy(sample, 0, temp, 0, (Size - 1) * Dim);
            }
            
            // Yeni veriyi ekle
            Array.Copy(x, 0, temp, (Size - 1) * Dim, Dim);
            return temp;
        }

        public static float[] Add_Labels(float[] Labels, int Size, int label)
        {
            float[] temp = new float[Size];
            
            // Mevcut etiketleri kopyala
            if (Size > 1)
            {
                Array.Copy(Labels, 0, temp, 0, Size - 1);
            }
            
            // Yeni etiketi ekle
            temp[Size - 1] = label;
            return temp;
        }

        public static float[] init_array_random(int len)
        {
            float[] arr = new float[len];
            Random rand = new Random();
            
            for (int i = 0; i < len; i++)
            {
                arr[i] = (float)(rand.NextDouble() - 0.5);
            }
            return arr;
        }

        public static void Z_Score_Parameters(float[] x, int Size, int dim, float[] mean, float[] std)
        {
            float[] Total = new float[dim];
            int i, j;

            // Initialize arrays
            for (i = 0; i < dim; i++)
            {
                mean[i] = 0.0f;
                std[i] = 0.0f;
                Total[i] = 0.0f;
            }

            // Calculate mean
            for (i = 0; i < Size; i++)
            {
                for (j = 0; j < dim; j++)
                {
                    Total[j] += x[i * dim + j];
                }
            }

            for (i = 0; i < dim; i++)
            {
                mean[i] = Total[i] / Size;
            }

            // Calculate standard deviation
            for (i = 0; i < Size; i++)
            {
                for (j = 0; j < dim; j++)
                {
                    float diff = x[i * dim + j] - mean[j];
                    std[j] += diff * diff;
                }
            }

            for (j = 0; j < dim; j++)
            {
                std[j] = (float)Math.Sqrt(std[j] / Size);
            }
        }

        public static int Test_Forward(float[] x, float[] weight, float[] bias, int num_Class, int inputDim)
        {
            int i, j, index_Max = 0;

            if (num_Class > 2)
            {
                float[] output = new float[num_Class];
                
                // Output layer calculation
                for (i = 0; i < num_Class; i++)
                {
                    output[i] = 0.0f;
                    for (j = 0; j < inputDim; j++)
                    {
                        output[i] += weight[i * inputDim + j] * x[j];
                    }
                    output[i] += bias[i];
                    output[i] = (float)Math.Tanh(output[i]);
                }

                // Find maximum output
                float temp = output[0];
                index_Max = 0;
                for (i = 1; i < num_Class; i++)
                {
                    if (temp < output[i])
                    {
                        temp = output[i];
                        index_Max = i;
                    }
                }
            }
            else
            {
                // Binary classification
                float output = 0.0f;
                for (j = 0; j < inputDim; j++)
                {
                    output += weight[j] * x[j];
                }
                output += bias[0];
                output = (float)Math.Tanh(output);
                
                index_Max = (output > 0.0f) ? 0 : 1;
            }

            return index_Max;
        }

        // Dosya okuma/yazma fonksiyonları
        public static bool ReadDataFromFile(string filename, out int dim, out int width, out int height, out int numClasses, out float[] samples, out float[] labels)
        {
            dim = 0; width = 0; height = 0; numClasses = 0;
            samples = null; labels = null;

            try
            {
                using (StreamReader file = new StreamReader(filename))
                {
                    string firstLine = file.ReadLine();
                    string[] firstValues = firstLine.Split(' ');
                    
                    if (firstValues.Length >= 4)
                    {
                        dim = int.Parse(firstValues[0]);
                        width = int.Parse(firstValues[1]);
                        height = int.Parse(firstValues[2]);
                        numClasses = int.Parse(firstValues[3]);
                    }

                    // Geri kalan satırları oku
                    var sampleList = new System.Collections.Generic.List<float>();
                    var labelList = new System.Collections.Generic.List<float>();
                    
                    string line;
                    while ((line = file.ReadLine()) != null)
                    {
                        string[] values = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        if (values.Length >= dim + 1)
                        {
                            for (int i = 0; i < dim; i++)
                            {
                                sampleList.Add(float.Parse(values[i]));
                            }
                            labelList.Add(float.Parse(values[dim]));
                        }
                    }

                    samples = sampleList.ToArray();
                    labels = labelList.ToArray();
                }
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine("File read error: " + ex.Message);
                return false;
            }
        }

        public static bool SaveDataToFile(string filename, int dim, int width, int height, int numClasses, float[] samples, float[] labels, int sampleCount)
        {
            try
            {
                // Data klasörünü oluştur
                string directory = Path.GetDirectoryName(filename);
                if (!Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                using (StreamWriter file = new StreamWriter(filename))
                {
                    // Header
                    file.WriteLine($"{dim} {width} {height} {numClasses}");

                    // Samples and labels
                    for (int i = 0; i < sampleCount; i++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            file.Write(samples[i * dim + d] + " ");
                        }
                        file.WriteLine(labels[i]);
                    }
                }
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine("File save error: " + ex.Message);
                return false;
            }
        }

        public static bool SaveWeightsToFile(string filename, int inputDim, int numClasses, float[] weights, float[] biases)
        {
            try
            {
                // Data klasörünü oluştur
                string directory = Path.GetDirectoryName(filename);
                if (!Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                using (StreamWriter file = new StreamWriter(filename))
                {
                    // Header: Layer Dimension numClass
                    file.WriteLine($"1 {inputDim} {numClasses}");

                    // Weights
                    for (int k = 0; k < weights.Length; k++)
                    {
                        file.Write(weights[k] + " ");
                    }
                    file.WriteLine();

                    // Biases
                    for (int k = 0; k < biases.Length; k++)
                    {
                        file.Write(biases[k] + " ");
                    }
                }
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Weights save error: " + ex.Message);
                return false;
            }
        }
    }
}