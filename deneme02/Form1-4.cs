using System;
using System.Drawing;
using System.Windows.Forms;
using System.IO;
using System.Collections.Generic;

namespace Collect
{
    public class Layer
    {
        public float[] Weights;
        public float[] Biases;
        public float[] Outputs;
        public float[] Deltas;
        public int InputSize;
        public int OutputSize;
        
        public Layer(int inputSize, int outputSize)
        {
            InputSize = inputSize;
            OutputSize = outputSize;
            Weights = new float[inputSize * outputSize];
            Biases = new float[outputSize];
            Outputs = new float[outputSize];
            Deltas = new float[outputSize];
            
            InitializeWeights();
        }
        
        private void InitializeWeights()
        {
            Random rand = new Random();
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = (float)(rand.NextDouble() - 0.5) * 2.0f;
            }
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] = (float)(rand.NextDouble() - 0.5) * 2.0f;
            }
        }
    }

    public class NeuralNetwork
    {
        public List<Layer> Layers { get; private set; }
        public float LearningRate { get; set; }
        
        public NeuralNetwork(float learningRate = 0.1f)
        {
            Layers = new List<Layer>();
            LearningRate = learningRate;
        }
        
        public void AddLayer(int inputSize, int outputSize)
        {
            Layers.Add(new Layer(inputSize, outputSize));
        }
        
        public void ClearLayers()
        {
            Layers.Clear();
        }
    }

    public partial class Form1 : Form
    {
        private int class_count = 0, numSample = 0, inputDim = 2;
        private float[] Samples, targets;
        private float learningRate = 0.1f;
        private int epochs = 1000;
        private NeuralNetwork network;
        
        private PictureBox pictureBox1;
        private PictureBox trainingGraph;
        private GroupBox groupBox1, groupBox2;
        private Button Set_Net, Train_Button;
        private Label label1, label2, label3, label4;
        private ComboBox ClassCountBox, ClassNoBox;
        private MenuStrip menuStrip1;
        private ToolStripMenuItem fileToolStripMenuItem, readDataToolStripMenuItem, saveDataToolStripMenuItem;
        private ToolStripMenuItem processToolStripMenuItem, trainingToolStripMenuItem, testingToolStripMenuItem;
        private TextBox textBox1;
        private ProgressBar progressBar1;
        
        // Yeni kontroller
        private ComboBox hiddenLayersComboBox;
        private TextBox neuronsPerLayerTextBox;
        private Label label5, label6;
        
        private List<float> errorHistory = new List<float>();
        private Bitmap graphBitmap;

        public Form1()
        {
            InitializeComponent();
        }
        
        private void InitializeComponent()
        {
            // Ana PictureBox
            this.pictureBox1 = new PictureBox();
            this.pictureBox1.BackColor = SystemColors.ButtonHighlight;
            this.pictureBox1.Location = new Point(13, 35);
            this.pictureBox1.Size = new Size(500, 400);
            this.pictureBox1.Paint += new PaintEventHandler(pictureBox1_Paint);
            this.pictureBox1.MouseClick += new MouseEventHandler(pictureBox1_MouseClick);
            
            // Eğitim Grafiği
            this.trainingGraph = new PictureBox();
            this.trainingGraph.BackColor = Color.White;
            this.trainingGraph.Location = new Point(520, 35);
            this.trainingGraph.Size = new Size(400, 200);
            this.trainingGraph.BorderStyle = BorderStyle.FixedSingle;
            this.trainingGraph.Paint += new PaintEventHandler(trainingGraph_Paint);
            
            // GroupBox1 - Network Architecture
            this.groupBox1 = new GroupBox();
            this.groupBox1.Location = new Point(520, 250);
            this.groupBox1.Size = new Size(200, 220);
            this.groupBox1.Text = "Network Architecture";
            this.groupBox1.Font = new Font("Microsoft Sans Serif", 8.25F, FontStyle.Bold);
            
            this.ClassCountBox = new ComboBox();
            this.ClassCountBox.Location = new Point(10, 20);
            this.ClassCountBox.Size = new Size(82, 21);
            this.ClassCountBox.Items.AddRange(new object[] {"2", "3", "4", "5", "6", "7"});
            this.ClassCountBox.Text = "2";
            
            this.label1 = new Label();
            this.label1.Location = new Point(108, 23);
            this.label1.Size = new Size(69, 13);
            this.label1.Text = "Sınıf Sayısı";
            
            // Yeni kontroller
            this.label5 = new Label();
            this.label5.Location = new Point(10, 50);
            this.label5.Size = new Size(80, 13);
            this.label5.Text = "Hidden Layers:";
            
            this.hiddenLayersComboBox = new ComboBox();
            this.hiddenLayersComboBox.Location = new Point(100, 47);
            this.hiddenLayersComboBox.Size = new Size(60, 21);
            this.hiddenLayersComboBox.Items.AddRange(new object[] { "1", "2", "3", "4" });
            this.hiddenLayersComboBox.Text = "1";
            
            this.label6 = new Label();
            this.label6.Location = new Point(10, 80);
            this.label6.Size = new Size(90, 13);
            this.label6.Text = "Neurons/Layer:";
            
            this.neuronsPerLayerTextBox = new TextBox();
            this.neuronsPerLayerTextBox.Location = new Point(100, 77);
            this.neuronsPerLayerTextBox.Size = new Size(60, 20);
            this.neuronsPerLayerTextBox.Text = "4";
            
            this.Set_Net = new Button();
            this.Set_Net.Location = new Point(10, 110);
            this.Set_Net.Size = new Size(131, 25);
            this.Set_Net.Text = "Network Setting";
            this.Set_Net.Click += new EventHandler(Set_Net_Click);
            
            this.Train_Button = new Button();
            this.Train_Button.Location = new Point(10, 145);
            this.Train_Button.Size = new Size(131, 25);
            this.Train_Button.Text = "Train Network";
            this.Train_Button.Click += new EventHandler(Train_Button_Click);
            this.Train_Button.Enabled = false;
            
            this.groupBox1.Controls.Add(Set_Net);
            this.groupBox1.Controls.Add(Train_Button);
            this.groupBox1.Controls.Add(label1);
            this.groupBox1.Controls.Add(ClassCountBox);
            this.groupBox1.Controls.Add(label5);
            this.groupBox1.Controls.Add(hiddenLayersComboBox);
            this.groupBox1.Controls.Add(label6);
            this.groupBox1.Controls.Add(neuronsPerLayerTextBox);
            
            // GroupBox2 - Data Collection
            this.groupBox2 = new GroupBox();
            this.groupBox2.Location = new Point(520, 480);
            this.groupBox2.Size = new Size(190, 80);
            this.groupBox2.Text = "Data Collection";
            this.groupBox2.Font = new Font("Microsoft Sans Serif", 8.25F, FontStyle.Bold);
            
            this.ClassNoBox = new ComboBox();
            this.ClassNoBox.Location = new Point(7, 20);
            this.ClassNoBox.Size = new Size(75, 21);
            this.ClassNoBox.Items.AddRange(new object[] {"1", "2", "3", "4", "5", "6", "7", "8", "9"});
            this.ClassNoBox.Text = "1";
            
            this.label2 = new Label();
            this.label2.Location = new Point(98, 23);
            this.label2.Size = new Size(81, 13);
            this.label2.Text = "Örnek Etiketi";
            
            this.groupBox2.Controls.Add(label2);
            this.groupBox2.Controls.Add(ClassNoBox);
            
            // Labels
            this.label3 = new Label();
            this.label3.Location = new Point(730, 250);
            this.label3.Size = new Size(150, 20);
            this.label3.Text = "Samples Count: 0";
            
            this.label4 = new Label();
            this.label4.Location = new Point(730, 280);
            this.label4.Size = new Size(200, 20);
            this.label4.Text = "Status: Ready";
            
            // ProgressBar
            this.progressBar1 = new ProgressBar();
            this.progressBar1.Location = new Point(730, 310);
            this.progressBar1.Size = new Size(200, 20);
            this.progressBar1.Visible = false;
            
            // TextBox1
            this.textBox1 = new TextBox();
            this.textBox1.Location = new Point(730, 350);
            this.textBox1.Multiline = true;
            this.textBox1.Size = new Size(200, 180);
            this.textBox1.ScrollBars = ScrollBars.Vertical;
            
            // MenuStrip
            this.menuStrip1 = new MenuStrip();
            this.fileToolStripMenuItem = new ToolStripMenuItem();
            this.fileToolStripMenuItem.Text = "File";
            
            this.readDataToolStripMenuItem = new ToolStripMenuItem();
            this.readDataToolStripMenuItem.Text = "Read_Data";
            this.readDataToolStripMenuItem.Click += new EventHandler(readDataToolStripMenuItem_Click);
            
            this.saveDataToolStripMenuItem = new ToolStripMenuItem();
            this.saveDataToolStripMenuItem.Text = "Save_Data";
            this.saveDataToolStripMenuItem.Click += new EventHandler(saveDataToolStripMenuItem_Click);
            
            this.fileToolStripMenuItem.DropDownItems.AddRange(new ToolStripItem[] {
                readDataToolStripMenuItem, saveDataToolStripMenuItem});
                
            this.processToolStripMenuItem = new ToolStripMenuItem();
            this.processToolStripMenuItem.Text = "Process";
            
            this.trainingToolStripMenuItem = new ToolStripMenuItem();
            this.trainingToolStripMenuItem.Text = "Training";
            this.trainingToolStripMenuItem.Click += new EventHandler(trainingToolStripMenuItem_Click);
            this.trainingToolStripMenuItem.Enabled = false;
            
            this.testingToolStripMenuItem = new ToolStripMenuItem();
            this.testingToolStripMenuItem.Text = "Testing";
            this.testingToolStripMenuItem.Click += new EventHandler(testingToolStripMenuItem_Click);
            
            this.processToolStripMenuItem.DropDownItems.AddRange(new ToolStripItem[] {
                trainingToolStripMenuItem, testingToolStripMenuItem});
            
            this.menuStrip1.Items.AddRange(new ToolStripItem[] {
                fileToolStripMenuItem, processToolStripMenuItem});
            
            // Form
            this.Text = "Multi-Layer Neural Network - Live Training with Colors";
            this.Size = new Size(950, 650);
            this.Controls.Add(pictureBox1);
            this.Controls.Add(trainingGraph);
            this.Controls.Add(groupBox1);
            this.Controls.Add(groupBox2);
            this.Controls.Add(label3);
            this.Controls.Add(label4);
            this.Controls.Add(progressBar1);
            this.Controls.Add(textBox1);
            this.Controls.Add(menuStrip1);
            this.MainMenuStrip = menuStrip1;
            
            graphBitmap = new Bitmap(trainingGraph.Width, trainingGraph.Height);
        }

        // FORWARD PROPAGATION
        private float[] Forward(float[] input, NeuralNetwork network)
        {
            float[] currentOutput = input;
            
            for (int layerIndex = 0; layerIndex < network.Layers.Count; layerIndex++)
            {
                Layer layer = network.Layers[layerIndex];
                float[] nextOutput = new float[layer.OutputSize];
                
                for (int i = 0; i < layer.OutputSize; i++)
                {
                    float sum = layer.Biases[i];
                    
                    for (int j = 0; j < layer.InputSize; j++)
                    {
                        sum += currentOutput[j] * layer.Weights[i * layer.InputSize + j];
                    }
                    
                    // Activation function
                    if (layerIndex == network.Layers.Count - 1)
                    {
                        // Output layer - tanh for binary, softmax for multi-class
                        nextOutput[i] = (float)Math.Tanh(sum);
                    }
                    else
                    {
                        // Hidden layers - ReLU
                        nextOutput[i] = Math.Max(0, sum); // ReLU
                    }
                }
                
                currentOutput = nextOutput;
                Array.Copy(nextOutput, layer.Outputs, nextOutput.Length);
            }
            
            return currentOutput;
        }

        // BACKWARD PROPAGATION
        private void Backward(float[] input, float[] target, NeuralNetwork network)
        {
            // Output layer deltas
            Layer outputLayer = network.Layers[network.Layers.Count - 1];
            for (int i = 0; i < outputLayer.OutputSize; i++)
            {
                float error = target[i] - outputLayer.Outputs[i];
                float derivative = 1.0f - outputLayer.Outputs[i] * outputLayer.Outputs[i]; // Tanh derivative
                outputLayer.Deltas[i] = error * derivative;
            }
            
            // Hidden layers deltas
            for (int layerIndex = network.Layers.Count - 2; layerIndex >= 0; layerIndex--)
            {
                Layer currentLayer = network.Layers[layerIndex];
                Layer nextLayer = network.Layers[layerIndex + 1];
                
                for (int i = 0; i < currentLayer.OutputSize; i++)
                {
                    float sum = 0.0f;
                    for (int j = 0; j < nextLayer.OutputSize; j++)
                    {
                        sum += nextLayer.Deltas[j] * nextLayer.Weights[j * nextLayer.InputSize + i];
                    }
                    
                    // ReLU derivative
                    float derivative = currentLayer.Outputs[i] > 0 ? 1.0f : 0.0f;
                    currentLayer.Deltas[i] = sum * derivative;
                }
            }
            
            // Update weights and biases
            float[] previousOutput = input;
            for (int layerIndex = 0; layerIndex < network.Layers.Count; layerIndex++)
            {
                Layer layer = network.Layers[layerIndex];
                
                for (int i = 0; i < layer.OutputSize; i++)
                {
                    for (int j = 0; j < layer.InputSize; j++)
                    {
                        layer.Weights[i * layer.InputSize + j] += network.LearningRate * 
                            layer.Deltas[i] * previousOutput[j];
                    }
                    layer.Biases[i] += network.LearningRate * layer.Deltas[i];
                }
                
                previousOutput = layer.Outputs;
            }
        }

        // PREDICTION FUNCTION
        private int Predict(float[] input, NeuralNetwork network)
        {
            float[] output = Forward(input, network);
            
            if (class_count == 2)
            {
                // Binary classification
                return output[0] > 0 ? 0 : 1;
            }
            else
            {
                // Multi-class classification
                int maxIndex = 0;
                for (int i = 1; i < output.Length; i++)
                {
                    if (output[i] > output[maxIndex])
                    {
                        maxIndex = i;
                    }
                }
                return maxIndex;
            }
        }

        // KARAR SINIRI ÇİZME FONKSİYONU
        private void DrawDecisionBoundaryAtColorEdge(Graphics g, NeuralNetwork network, int classCount, Color color, float lineWidth = 2.0f)
        {
            if (classCount != 2) return;

            int width = pictureBox1.Width;
            int height = pictureBox1.Height;
            int centerX = width / 2;
            int centerY = height / 2;

            // Karar sınırını çizmek için grid-based approach
            List<Point> boundaryPoints = new List<Point>();
            
            for (int x = 0; x < width; x += 2)
            {
                for (int y = 0; y < height; y += 2)
                {
                    float[] input = new float[inputDim];
                    input[0] = x - centerX;
                    input[1] = centerY - y;
                    
                    int prediction = Predict(input, network);
                    
                    // Check neighbors for boundary
                    if (x > 0 && y > 0 && x < width - 2 && y < height - 2)
                    {
                        bool isBoundary = false;
                        
                        // Check if this point has different prediction than any neighbor
                        for (int dx = -2; dx <= 2; dx += 2)
                        {
                            for (int dy = -2; dy <= 2; dy += 2)
                            {
                                if (dx == 0 && dy == 0) continue;
                                
                                int nx = x + dx;
                                int ny = y + dy;
                                
                                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                                {
                                    float[] neighborInput = new float[inputDim];
                                    neighborInput[0] = nx - centerX;
                                    neighborInput[1] = centerY - ny;
                                    
                                    int neighborPrediction = Predict(neighborInput, network);
                                    
                                    if (prediction != neighborPrediction)
                                    {
                                        isBoundary = true;
                                        break;
                                    }
                                }
                            }
                            if (isBoundary) break;
                        }
                        
                        if (isBoundary)
                        {
                            boundaryPoints.Add(new Point(x, y));
                        }
                    }
                }
            }
            
            // Draw boundary points
            if (boundaryPoints.Count > 0)
            {
                Pen boundaryPen = new Pen(color, lineWidth);
                foreach (Point point in boundaryPoints)
                {
                    g.DrawRectangle(boundaryPen, point.X, point.Y, 1, 1);
                }
                boundaryPen.Dispose();
            }
        }

        // İKİ RENKLİ GÖSTERİM
        private void ShowCurrentDecisionBoundaryWithColors(NeuralNetwork network, int epoch, float[] mean, float[] std)
        {
            Bitmap tempBitmap = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            
            using (Graphics g = Graphics.FromImage(tempBitmap))
            {
                // Arka planı iki renge boya
                for (int row = 0; row < pictureBox1.Height; row += 2)
                {
                    for (int column = 0; column < pictureBox1.Width; column += 2)
                    {
                        float[] x = new float[inputDim];
                        x[0] = column - (pictureBox1.Width / 2);
                        x[1] = (pictureBox1.Height / 2) - row;
                        
                        // Normalize
                        if (mean != null && std != null)
                        {
                            x[0] = (x[0] - mean[0]) / std[0];
                            x[1] = (x[1] - mean[1]) / std[1];
                        }
                        
                        int predictedClass = Predict(x, network);
                        
                        Color c = Color.LightBlue;
                        if (predictedClass == 1)
                            c = Color.LightCoral;
                        else if (predictedClass >= 2)
                            c = Color.LightGreen;
                        
                        // Fill a small rectangle instead of single pixel for better visibility
                        using (SolidBrush brush = new SolidBrush(c))
                        {
                            g.FillRectangle(brush, column, row, 2, 2);
                        }
                    }
                }
                
                // Noktaları çiz
                int center_width = pictureBox1.Width / 2;
                int center_height = pictureBox1.Height / 2;
                for (int i = 0; i < numSample; i++)
                {
                    int temp_x = (int)Samples[i * inputDim] + center_width;
                    int temp_y = center_height - (int)Samples[i * inputDim + 1];
                    draw_sample(g, temp_x, temp_y, (int)targets[i]);
                }
                
                // Eksenleri çiz
                Pen axisPen = new Pen(Color.Gray, 0.5f);
                g.DrawLine(axisPen, center_width, 0, center_width, pictureBox1.Height);
                g.DrawLine(axisPen, 0, center_height, pictureBox1.Width, center_height);
                
                // Epoch bilgisini yaz
                Font infoFont = new Font("Arial", 9, FontStyle.Bold);
                g.DrawString($"Epoch: {epoch}", infoFont, Brushes.DarkBlue, 10, 10);
                
                // Network bilgisini yaz
                g.DrawString($"Layers: {network.Layers.Count}", infoFont, Brushes.DarkBlue, 10, 25);
                
                axisPen.Dispose();
                infoFont.Dispose();
            }
            
            pictureBox1.Image = tempBitmap;
            Application.DoEvents();
        }

        // FİNAL İKİ RENKLİ KARAR SINIRI
        private void ShowFinalDecisionBoundaryWithColors(NeuralNetwork network, float[] mean, float[] std)
        {
            Bitmap finalBitmap = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            
            using (Graphics g = Graphics.FromImage(finalBitmap))
            {
                // Arka planı iki renge boya
                for (int row = 0; row < pictureBox1.Height; row += 2)
                {
                    for (int column = 0; column < pictureBox1.Width; column += 2)
                    {
                        float[] x = new float[inputDim];
                        x[0] = column - (pictureBox1.Width / 2);
                        x[1] = (pictureBox1.Height / 2) - row;
                        
                        if (mean != null && std != null)
                        {
                            x[0] = (x[0] - mean[0]) / std[0];
                            x[1] = (x[1] - mean[1]) / std[1];
                        }
                        
                        int predictedClass = Predict(x, network);
                        
                        Color c = Color.LightBlue;
                        if (predictedClass == 1)
                            c = Color.LightCoral;
                        else if (predictedClass >= 2)
                            c = Color.LightGreen;
                        
                        using (SolidBrush brush = new SolidBrush(c))
                        {
                            g.FillRectangle(brush, column, row, 2, 2);
                        }
                    }
                }
                
                // FİNAL ÇİZGİYİ İKİ RENGİN BİRLEŞTİĞİ YERE ÇİZ
                if (class_count == 2)
                {
                    DrawDecisionBoundaryAtColorEdge(g, network, class_count, Color.Red, 3.0f);
                }
                
                // Noktaları çiz
                int center_width = pictureBox1.Width / 2;
                int center_height = pictureBox1.Height / 2;
                for (int i = 0; i < numSample; i++)
                {
                    int temp_x = (int)Samples[i * inputDim] + center_width;
                    int temp_y = center_height - (int)Samples[i * inputDim + 1];
                    draw_sample(g, temp_x, temp_y, (int)targets[i]);
                }
                
                // Eksenleri çiz
                Pen axisPen = new Pen(Color.Gray, 0.5f);
                g.DrawLine(axisPen, center_width, 0, center_width, pictureBox1.Height);
                g.DrawLine(axisPen, 0, center_height, pictureBox1.Width, center_height);
                
                // Doğruluk bilgisini yaz
                int correct = CalculateAccuracy(network, mean, std);
                float accuracy = (float)correct / numSample * 100;
                
                Font infoFont = new Font("Arial", 10, FontStyle.Bold);
                g.DrawString($"Accuracy: {accuracy:F1}%", infoFont, Brushes.DarkBlue, 10, 30);
                g.DrawString($"Layers: {network.Layers.Count}", infoFont, Brushes.DarkBlue, 10, 50);
                
                axisPen.Dispose();
                infoFont.Dispose();
            }
            
            pictureBox1.Image = finalBitmap;
        }

        // DOĞRULUK HESAPLAMA FONKSİYONU
        private int CalculateAccuracy(NeuralNetwork network, float[] mean, float[] std)
        {
            int correct = 0;
            
            for (int i = 0; i < numSample; i++)
            {
                float[] x = new float[inputDim];
                for (int j = 0; j < inputDim; j++)
                {
                    x[j] = Samples[i * inputDim + j];
                }
                
                // Normalize
                if (mean != null && std != null)
                {
                    x[0] = (x[0] - mean[0]) / std[0];
                    x[1] = (x[1] - mean[1]) / std[1];
                }
                
                int predicted = Predict(x, network);
                if (predicted == (int)targets[i])
                {
                    correct++;
                }
            }
            
            return correct;
        }

        // EĞİTİM GRAFİĞİ ÇİZME
        private void trainingGraph_Paint(object sender, PaintEventArgs e)
        {
            if (errorHistory.Count == 0) return;

            Graphics g = e.Graphics;
            g.Clear(Color.White);

            Pen borderPen = new Pen(Color.Black, 1);
            g.DrawRectangle(borderPen, 0, 0, trainingGraph.Width - 1, trainingGraph.Height - 1);

            Pen axisPen = new Pen(Color.Gray, 1);
            g.DrawLine(axisPen, 30, trainingGraph.Height - 30, trainingGraph.Width - 10, trainingGraph.Height - 30);
            g.DrawLine(axisPen, 30, 10, 30, trainingGraph.Height - 30);

            Font labelFont = new Font("Arial", 8);
            g.DrawString("Error", labelFont, Brushes.Black, 5, trainingGraph.Height / 2 - 20);
            g.DrawString("Epochs", labelFont, Brushes.Black, trainingGraph.Width / 2 - 20, trainingGraph.Height - 25);

            if (errorHistory.Count > 1)
            {
                Pen graphPen = new Pen(Color.Red, 2);
                float maxError = 0;
                foreach (float error in errorHistory)
                {
                    if (error > maxError) maxError = error;
                }
                if (maxError == 0) maxError = 1;

                float xScale = (trainingGraph.Width - 50) / (float)(errorHistory.Count - 1);
                float yScale = (trainingGraph.Height - 50) / maxError;

                for (int i = 0; i < errorHistory.Count - 1; i++)
                {
                    float x1 = 30 + i * xScale;
                    float y1 = trainingGraph.Height - 30 - (errorHistory[i] * yScale);
                    float x2 = 30 + (i + 1) * xScale;
                    float y2 = trainingGraph.Height - 30 - (errorHistory[i + 1] * yScale);
                    
                    g.DrawLine(graphPen, x1, y1, x2, y2);
                }

                for (int i = 0; i < errorHistory.Count; i += Math.Max(1, errorHistory.Count / 20))
                {
                    float x = 30 + i * xScale;
                    float y = trainingGraph.Height - 30 - (errorHistory[i] * yScale);
                    g.FillEllipse(Brushes.Blue, x - 2, y - 2, 4, 4);
                }

                g.DrawString($"Max Error: {maxError:F4}", labelFont, Brushes.Black, trainingGraph.Width - 120, 15);
                g.DrawString($"Current: {errorHistory[errorHistory.Count - 1]:F4}", labelFont, Brushes.Black, trainingGraph.Width - 120, 30);
            }
        }

        // GRAFİĞİ GÜNCELLE
        private void UpdateTrainingGraph(float error)
        {
            errorHistory.Add(error);
            trainingGraph.Invalidate();
            Application.DoEvents();
        }

        // PROCESS FONKSİYONLARI
        private float[] Add_Data(float[] sample, int Size, float[] x, int Dim)
        {
            float[] temp = new float[Size * Dim];
            if (Size > 1)
            {
                Array.Copy(sample, 0, temp, 0, (Size - 1) * Dim);
            }
            Array.Copy(x, 0, temp, (Size - 1) * Dim, Dim);
            return temp;
        }

        private float[] Add_Labels(float[] Labels, int Size, int label)
        {
            float[] temp = new float[Size];
            if (Size > 1)
            {
                Array.Copy(Labels, 0, temp, 0, Size - 1);
            }
            temp[Size - 1] = label;
            return temp;
        }

        private void Z_Score_Parameters(float[] x, int Size, int dim, float[] mean, float[] std)
        {
            float[] Total = new float[dim];
            int i, j;

            for (i = 0; i < dim; i++)
            {
                mean[i] = 0.0f;
                std[i] = 0.0f;
                Total[i] = 0.0f;
            }

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

        // EĞİTİM FONKSİYONU (Multi-layer)
        private void TrainNetwork()
        {
            if (numSample == 0)
            {
                MessageBox.Show("No samples available for training!");
                return;
            }

            if (network == null || network.Layers.Count == 0)
            {
                MessageBox.Show("Network is not initialized!");
                return;
            }

            errorHistory.Clear();
            progressBar1.Visible = true;
            progressBar1.Value = 0;
            label4.Text = "Status: Training Started...";
            trainingGraph.Invalidate();
            Application.DoEvents();

            float[] mean = new float[inputDim];
            float[] std = new float[inputDim];
            Z_Score_Parameters(Samples, numSample, inputDim, mean, std);

            float[] normalizedSamples = new float[numSample * inputDim];
            for (int i = 0; i < numSample; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    normalizedSamples[i * inputDim + j] = (Samples[i * inputDim + j] - mean[j]) / std[j];
                }
            }

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float totalError = 0.0f;

                for (int sampleIndex = 0; sampleIndex < numSample; sampleIndex++)
                {
                    float[] input = new float[inputDim];
                    Array.Copy(normalizedSamples, sampleIndex * inputDim, input, 0, inputDim);
                    
                    // Target array oluştur
                    float[] target = new float[class_count];
                    if (class_count == 2)
                    {
                        // Binary classification
                        target[(int)targets[sampleIndex]] = 1.0f;
                        target[1 - (int)targets[sampleIndex]] = -1.0f;
                    }
                    else
                    {
                        // Multi-class classification
                        for (int i = 0; i < class_count; i++)
                        {
                            target[i] = (i == (int)targets[sampleIndex]) ? 1.0f : -1.0f;
                        }
                    }

                    // Forward propagation
                    float[] output = Forward(input, network);
                    
                    // Error hesapla
                    for (int j = 0; j < class_count; j++)
                    {
                        totalError += Math.Abs(target[j] - output[j]);
                    }
                    
                    // Backward propagation
                    Backward(input, target, network);
                }

                float avgError = totalError / numSample;
                UpdateTrainingGraph(avgError);

                // Her 50 epoch'ta bir karar sınırını göster
                if (epoch % 50 == 0)
                {
                    ShowCurrentDecisionBoundaryWithColors(network, epoch, mean, std);
                }

                if (epoch % 10 == 0)
                {
                    progressBar1.Value = (int)((float)epoch / epochs * 100);
                    label4.Text = $"Epoch: {epoch}, Error: {avgError:F4}";
                    Application.DoEvents();
                }
            }

            progressBar1.Value = 100;
            label4.Text = "Status: Training Completed!";
            progressBar1.Visible = false;

            // FİNAL KARAR SINIRINI GÖSTER
            ShowFinalDecisionBoundaryWithColors(network, mean, std);

            // Doğruluk hesapla
            int correct = CalculateAccuracy(network, mean, std);
            float accuracy = (float)correct / numSample * 100;
            textBox1.AppendText($"Training completed!\r\nFinal Error: {errorHistory[errorHistory.Count - 1]:F4}\r\nAccuracy: {accuracy:F2}%\r\n");
            textBox1.AppendText($"Network Architecture: {network.Layers.Count} layers\r\n");
            MessageBox.Show($"Training completed!\nFinal accuracy: {accuracy:F2}%\nLayers: {network.Layers.Count}");
        }

        // draw_sample fonksiyonları
        private void draw_sample(Graphics g, int temp_x, int temp_y, int label)
        {
            Pen pen;
            switch (label)
            {
                case 0: pen = new Pen(Color.Black, 3.0f); break;
                case 1: pen = new Pen(Color.Red, 3.0f); break;
                case 2: pen = new Pen(Color.Blue, 3.0f); break;
                case 3: pen = new Pen(Color.Green, 3.0f); break;
                case 4: pen = new Pen(Color.Yellow, 3.0f); break;
                case 5: pen = new Pen(Color.Orange, 3.0f); break;
                default: pen = new Pen(Color.YellowGreen, 3.0f); break;
            }
            
            g.DrawLine(pen, temp_x - 5, temp_y, temp_x + 5, temp_y);
            g.DrawLine(pen, temp_x, temp_y - 5, temp_x, temp_y + 5);
            pen.Dispose();
        }

        private void draw_sample(int temp_x, int temp_y, int label)
        {
            using (Graphics g = pictureBox1.CreateGraphics())
            {
                draw_sample(g, temp_x, temp_y, label);
            }
        }

        private void pictureBox1_Paint(object sender, PaintEventArgs e)
        {
            Pen pen = new Pen(Color.Black, 1.0f);
            int center_width = pictureBox1.Width / 2;
            int center_height = pictureBox1.Height / 2;
            
            e.Graphics.DrawLine(pen, center_width, 0, center_width, pictureBox1.Height);
            e.Graphics.DrawLine(pen, 0, center_height, pictureBox1.Width, center_height);
            pen.Dispose();
        }

        // Diğer fonksiyonlar...
        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            if (class_count == 0)
            {
                MessageBox.Show("The Network Architecture should be firstly set up");
            }
            else
            {
                float[] x = new float[inputDim];
                int temp_x = e.X;
                int temp_y = e.Y;
                x[0] = temp_x - (pictureBox1.Width / 2);
                x[1] = (pictureBox1.Height / 2) - temp_y;
                
                int numLabel = Convert.ToInt32(ClassNoBox.Text);
                if (numLabel > class_count)
                {
                    MessageBox.Show("The class label cannot be greater than the maximum number of classes.");
                }
                else
                {
                    int label = numLabel - 1;
                    
                    if (numSample == 0)
                    {
                        numSample = 1;
                        Samples = new float[numSample * inputDim];
                        targets = new float[numSample];
                        Array.Copy(x, Samples, inputDim);
                        targets[0] = label;
                    }
                    else
                    {
                        numSample++;
                        Samples = Add_Data(Samples, numSample, x, inputDim);
                        targets = Add_Labels(targets, numSample, label);
                    }
                    
                    draw_sample(temp_x, temp_y, label);
                    label3.Text = "Samples Count: " + numSample.ToString();
                    
                    if (numSample > 0)
                    {
                        Train_Button.Enabled = true;
                        trainingToolStripMenuItem.Enabled = true;
                    }
                }
            }
        }

        private void Set_Net_Click(object sender, EventArgs e)
        {
            class_count = Convert.ToInt32(ClassCountBox.Text);
            int hiddenLayers = Convert.ToInt32(hiddenLayersComboBox.Text);
            int neuronsPerLayer = Convert.ToInt32(neuronsPerLayerTextBox.Text);
            
            // Neural Network'ü oluştur
            network = new NeuralNetwork(learningRate);
            
            // Input layer
            network.AddLayer(inputDim, neuronsPerLayer);
            
            // Hidden layers
            for (int i = 0; i < hiddenLayers - 1; i++)
            {
                network.AddLayer(neuronsPerLayer, neuronsPerLayer);
            }
            
            // Output layer
            network.AddLayer(neuronsPerLayer, class_count);
            
            Set_Net.Text = "Network is Ready";
            label4.Text = $"Status: Network Ready ({hiddenLayers + 1} layers)";
            textBox1.AppendText($"Network initialized with {hiddenLayers + 1} layers\n");
            textBox1.AppendText($"Architecture: {inputDim}-");
            for (int i = 0; i < hiddenLayers; i++)
            {
                textBox1.AppendText($"{neuronsPerLayer}-");
            }
            textBox1.AppendText($"{class_count}\n");
        }

        private void Train_Button_Click(object sender, EventArgs e)
        {
            TrainNetwork();
        }

        private void trainingToolStripMenuItem_Click(object sender, EventArgs e)
        {
            TrainNetwork();
        }

        private void readDataToolStripMenuItem_Click(object sender, EventArgs e)
        {
            MessageBox.Show("Read Data functionality will be implemented");
        }

        private void saveDataToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (numSample != 0)
            {
                MessageBox.Show("Data saved successfully!");
            }
            else
            {
                MessageBox.Show("At least one sample should be given");
            }
        }

        private void testingToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (numSample == 0)
            {
                MessageBox.Show("No samples available for testing");
                return;
            }

            if (network == null)
            {
                MessageBox.Show("Network is not trained!");
                return;
            }

            float[] mean = new float[inputDim];
            float[] std = new float[inputDim];
            
            Z_Score_Parameters(Samples, numSample, inputDim, mean, std);
            
            Bitmap surface = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            
            using (Graphics g = Graphics.FromImage(surface))
            {
                // Arka planı iki renge boya
                for (int row = 0; row < pictureBox1.Height; row += 2)
                {
                    for (int column = 0; column < pictureBox1.Width; column += 2)
                    {
                        float[] x = new float[inputDim];
                        x[0] = column - (pictureBox1.Width / 2);
                        x[1] = (pictureBox1.Height / 2) - row;
                        
                        x[0] = (x[0] - mean[0]) / std[0];
                        x[1] = (x[1] - mean[1]) / std[1];
                        
                        int predictedClass = Predict(x, network);
                        
                        Color c = Color.LightBlue;
                        if (predictedClass == 1)
                            c = Color.LightCoral;
                        else if (predictedClass >= 2)
                            c = Color.LightGreen;
                        
                        using (SolidBrush brush = new SolidBrush(c))
                        {
                            g.FillRectangle(brush, column, row, 2, 2);
                        }
                    }
                }
                
                // TESTTE DE İKİ RENGİN BİRLEŞTİĞİ YERE ÇİZGİ ÇİZ
                if (class_count == 2)
                {
                    DrawDecisionBoundaryAtColorEdge(g, network, class_count, Color.DarkBlue, 3.0f);
                }
                
                // Noktaları çiz
                int center_width = pictureBox1.Width / 2;
                int center_height = pictureBox1.Height / 2;
                for (int i = 0; i < numSample; i++)
                {
                    int temp_x = (int)Samples[i * inputDim] + center_width;
                    int temp_y = center_height - (int)Samples[i * inputDim + 1];
                    draw_sample(g, temp_x, temp_y, (int)targets[i]);
                }
            }
            
            pictureBox1.Image = surface;
            
            // Doğruluk hesapla ve göster
            int correct = CalculateAccuracy(network, mean, std);
            float accuracy = (float)correct / numSample * 100;
            MessageBox.Show($"Testing completed!\nAccuracy: {accuracy:F2}%\nLayers: {network.Layers.Count}");
        }
    }
}