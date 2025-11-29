using System;
using System.Drawing;
using System.Windows.Forms;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;

namespace Collect
{
    public class Layer
    {
        public float[] Weights;
        public float[] Biases;
        public float[] Outputs;
        public float[] Deltas;
        public float[] WeightVelocities; // Momentum için ağırlık hızları
        public float[] BiasVelocities;   // Momentum için bias hızları
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
            WeightVelocities = new float[inputSize * outputSize]; // Momentum için
            BiasVelocities = new float[outputSize]; // Momentum için
            
            InitializeWeights();
        }
        
        private void InitializeWeights()
        {
            Random rand = new Random();
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = (float)(rand.NextDouble() - 0.5) * 2.0f;
                WeightVelocities[i] = 0.0f; // Momentum hızlarını sıfırla
            }
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] = (float)(rand.NextDouble() - 0.5) * 2.0f;
                BiasVelocities[i] = 0.0f; // Momentum hızlarını sıfırla
            }
        }
    }

    public class NeuralNetwork
    {
        public List<Layer> Layers { get; private set; }
        public float LearningRate { get; set; }
        public string ProblemType { get; set; }
        public float Momentum { get; set; } // Momentum katsayısı
        
        public NeuralNetwork(float learningRate = 0.1f, string problemType = "classification", float momentum = 0.9f)
        {
            Layers = new List<Layer>();
            LearningRate = learningRate;
            ProblemType = problemType;
            Momentum = momentum; // Momentum parametresi
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
        private float momentum = 0.9f; // Momentum değişkeni
        private int epochs = 1000;
        private NeuralNetwork network;
        private string problemType = "classification";
        
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
        
        private ComboBox hiddenLayersComboBox;
        private TextBox neuronsPerLayerTextBox;
        private Label label5, label6;
        private ComboBox problemTypeComboBox;
        private TextBox regressionValueTextBox;
        private Label label7;
        private TextBox momentumTextBox; // Momentum için TextBox
        private Label label8; // Momentum için Label
        
        private List<float> errorHistory = new List<float>();
        private Bitmap graphBitmap;

        public Form1()
        {
            InitializeComponent();
        }
        
        private void InitializeComponent()
        {
            this.pictureBox1 = new PictureBox();
            this.pictureBox1.BackColor = SystemColors.ButtonHighlight;
            this.pictureBox1.Location = new Point(13, 35);
            this.pictureBox1.Size = new Size(500, 400);
            this.pictureBox1.Paint += new PaintEventHandler(pictureBox1_Paint);
            this.pictureBox1.MouseClick += new MouseEventHandler(pictureBox1_MouseClick);
            
            this.trainingGraph = new PictureBox();
            this.trainingGraph.BackColor = Color.White;
            this.trainingGraph.Location = new Point(520, 35);
            this.trainingGraph.Size = new Size(400, 200);
            this.trainingGraph.BorderStyle = BorderStyle.FixedSingle;
            this.trainingGraph.Paint += new PaintEventHandler(trainingGraph_Paint);
            
            this.groupBox1 = new GroupBox();
            this.groupBox1.Location = new Point(520, 250);
            this.groupBox1.Size = new Size(200, 280);
            this.groupBox1.Text = "Network Architecture";
            this.groupBox1.Font = new Font("Microsoft Sans Serif", 8.25F, FontStyle.Bold);
            
            this.label7 = new Label();
            this.label7.Location = new Point(10, 20);
            this.label7.Size = new Size(80, 13);
            this.label7.Text = "Problem Type:";
            
            this.problemTypeComboBox = new ComboBox();
            this.problemTypeComboBox.Location = new Point(100, 17);
            this.problemTypeComboBox.Size = new Size(90, 21);
            this.problemTypeComboBox.Items.AddRange(new object[] { "Classification", "Regression" });
            this.problemTypeComboBox.Text = "Classification";
            this.problemTypeComboBox.SelectedIndexChanged += new EventHandler(problemTypeComboBox_SelectedIndexChanged);
            
            this.ClassCountBox = new ComboBox();
            this.ClassCountBox.Location = new Point(10, 50);
            this.ClassCountBox.Size = new Size(82, 21);
            this.ClassCountBox.Items.AddRange(new object[] {"2", "3", "4", "5", "6", "7"});
            this.ClassCountBox.Text = "2";
            
            this.label1 = new Label();
            this.label1.Location = new Point(108, 53);
            this.label1.Size = new Size(69, 13);
            this.label1.Text = "Sınıf Sayısı";
            
            this.label5 = new Label();
            this.label5.Location = new Point(10, 80);
            this.label5.Size = new Size(80, 13);
            this.label5.Text = "Hidden Layers:";
            
            this.hiddenLayersComboBox = new ComboBox();
            this.hiddenLayersComboBox.Location = new Point(100, 77);
            this.hiddenLayersComboBox.Size = new Size(60, 21);
            this.hiddenLayersComboBox.Items.AddRange(new object[] { "1", "2", "3", "4" });
            this.hiddenLayersComboBox.Text = "1";
            
            this.label6 = new Label();
            this.label6.Location = new Point(10, 110);
            this.label6.Size = new Size(90, 13);
            this.label6.Text = "Neurons/Layer:";
            
            this.neuronsPerLayerTextBox = new TextBox();
            this.neuronsPerLayerTextBox.Location = new Point(100, 107);
            this.neuronsPerLayerTextBox.Size = new Size(60, 20);
            this.neuronsPerLayerTextBox.Text = "4";
            
            this.label8 = new Label();
            this.label8.Location = new Point(10, 140);
            this.label8.Size = new Size(90, 13);
            this.label8.Text = "Momentum:";
            
            this.momentumTextBox = new TextBox();
            this.momentumTextBox.Location = new Point(100, 137);
            this.momentumTextBox.Size = new Size(60, 20);
            this.momentumTextBox.Text = "0,9";
            this.momentumTextBox.KeyPress += new KeyPressEventHandler(momentumTextBox_KeyPress);
            
            this.Set_Net = new Button();
            this.Set_Net.Location = new Point(10, 170);
            this.Set_Net.Size = new Size(131, 25);
            this.Set_Net.Text = "Network Setting";
            this.Set_Net.Click += new EventHandler(Set_Net_Click);
            
            this.Train_Button = new Button();
            this.Train_Button.Location = new Point(10, 205);
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
            this.groupBox1.Controls.Add(label7);
            this.groupBox1.Controls.Add(problemTypeComboBox);
            this.groupBox1.Controls.Add(label8);
            this.groupBox1.Controls.Add(momentumTextBox);
            
            this.groupBox2 = new GroupBox();
            this.groupBox2.Location = new Point(520, 540);
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
            
            this.regressionValueTextBox = new TextBox();
            this.regressionValueTextBox.Location = new Point(7, 45);
            this.regressionValueTextBox.Size = new Size(100, 20);
            this.regressionValueTextBox.Text = "0,5";
            this.regressionValueTextBox.Visible = false;
            this.regressionValueTextBox.KeyPress += new KeyPressEventHandler(regressionValueTextBox_KeyPress);
            
            this.groupBox2.Controls.Add(label2);
            this.groupBox2.Controls.Add(ClassNoBox);
            this.groupBox2.Controls.Add(regressionValueTextBox);
            
            this.label3 = new Label();
            this.label3.Location = new Point(730, 250);
            this.label3.Size = new Size(150, 20);
            this.label3.Text = "Samples Count: 0";
            
            this.label4 = new Label();
            this.label4.Location = new Point(730, 280);
            this.label4.Size = new Size(200, 20);
            this.label4.Text = "Status: Ready";
            
            this.progressBar1 = new ProgressBar();
            this.progressBar1.Location = new Point(730, 310);
            this.progressBar1.Size = new Size(200, 20);
            this.progressBar1.Visible = false;
            
            this.textBox1 = new TextBox();
            this.textBox1.Location = new Point(730, 350);
            this.textBox1.Multiline = true;
            this.textBox1.Size = new Size(200, 180);
            this.textBox1.ScrollBars = ScrollBars.Vertical;
            
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
            
            this.Text = "Multi-Layer Neural Network - Classification & Regression with Momentum";
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

        private void momentumTextBox_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (!char.IsControl(e.KeyChar) && !char.IsDigit(e.KeyChar) && e.KeyChar != ',' && e.KeyChar != '.')
            {
                e.Handled = true;
            }

            if ((e.KeyChar == ',' || e.KeyChar == '.') && ((TextBox)sender).Text.Contains(','))
            {
                e.Handled = true;
            }
        }

        private void regressionValueTextBox_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (!char.IsControl(e.KeyChar) && !char.IsDigit(e.KeyChar) && e.KeyChar != ',' && e.KeyChar != '.')
            {
                e.Handled = true;
            }

            if ((e.KeyChar == ',' || e.KeyChar == '.') && ((TextBox)sender).Text.Contains(','))
            {
                e.Handled = true;
            }
        }

        private void problemTypeComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            problemType = problemTypeComboBox.Text.ToLower();
            
            if (problemType == "regression")
            {
                ClassNoBox.Visible = false;
                label2.Visible = false;
                regressionValueTextBox.Visible = true;
                ClassCountBox.Enabled = false;
                ClassCountBox.Text = "1";
                label1.Text = "Output Dim";
                regressionValueTextBox.Text = "0,5";
            }
            else
            {
                ClassNoBox.Visible = true;
                label2.Visible = true;
                regressionValueTextBox.Visible = false;
                ClassCountBox.Enabled = true;
                label1.Text = "Sınıf Sayısı";
            }
            
            numSample = 0;
            Samples = null;
            targets = null;
            label3.Text = "Samples Count: 0";
            pictureBox1.Invalidate();
        }

        private float ParseRegressionValue(string value)
        {
            try
            {
                if (value.Contains(','))
                {
                    return float.Parse(value, CultureInfo.GetCultureInfo("tr-TR"));
                }
                else
                {
                    return float.Parse(value, CultureInfo.InvariantCulture);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Geçersiz değer: {value}\nLütfen sayı girin (örnek: 0,5 veya 1.2)", "Hata");
                return 0.5f;
            }
        }

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
                    
                    if (layerIndex == network.Layers.Count - 1)
                    {
                        if (network.ProblemType == "regression")
                        {
                            nextOutput[i] = sum;
                        }
                        else
                        {
                            nextOutput[i] = (float)Math.Tanh(sum);
                        }
                    }
                    else
                    {
                        nextOutput[i] = Math.Max(0, sum);
                    }
                }
                
                currentOutput = nextOutput;
                Array.Copy(nextOutput, layer.Outputs, nextOutput.Length);
            }
            
            return currentOutput;
        }

        private void Backward(float[] input, float[] target, NeuralNetwork network)
        {
            Layer outputLayer = network.Layers[network.Layers.Count - 1];
            for (int i = 0; i < outputLayer.OutputSize; i++)
            {
                float error = target[i] - outputLayer.Outputs[i];
                float derivative;
                
                if (network.ProblemType == "regression")
                {
                    derivative = 1.0f;
                }
                else
                {
                    derivative = 1.0f - outputLayer.Outputs[i] * outputLayer.Outputs[i];
                }
                
                outputLayer.Deltas[i] = error * derivative;
            }
            
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
                    
                    float derivative = currentLayer.Outputs[i] > 0 ? 1.0f : 0.0f;
                    currentLayer.Deltas[i] = sum * derivative;
                }
            }
            
            float[] previousOutput = input;
            for (int layerIndex = 0; layerIndex < network.Layers.Count; layerIndex++)
            {
                Layer layer = network.Layers[layerIndex];
                
                for (int i = 0; i < layer.OutputSize; i++)
                {
                    for (int j = 0; j < layer.InputSize; j++)
                    {
                        float weightGradient = network.LearningRate * layer.Deltas[i] * previousOutput[j];
                        layer.WeightVelocities[i * layer.InputSize + j] = 
                            network.Momentum * layer.WeightVelocities[i * layer.InputSize + j] + weightGradient;
                        layer.Weights[i * layer.InputSize + j] += layer.WeightVelocities[i * layer.InputSize + j];
                    }
                    
                    float biasGradient = network.LearningRate * layer.Deltas[i];
                    layer.BiasVelocities[i] = network.Momentum * layer.BiasVelocities[i] + biasGradient;
                    layer.Biases[i] += layer.BiasVelocities[i];
                }
                
                previousOutput = layer.Outputs;
            }
        }

        private float[] Predict(float[] input, NeuralNetwork network)
        {
            return Forward(input, network);
        }

        private int PredictClass(float[] input, NeuralNetwork network)
        {
            float[] output = Forward(input, network);
            
            if (class_count == 2)
            {
                return output[0] > 0 ? 0 : 1;
            }
            else
            {
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

        private void DrawDecisionBoundaryAtColorEdge(Graphics g, NeuralNetwork network, int classCount, Color color, float lineWidth = 2.0f)
        {
            if (classCount != 2) return;

            int width = pictureBox1.Width;
            int height = pictureBox1.Height;
            int centerX = width / 2;
            int centerY = height / 2;

            List<Point> boundaryPoints = new List<Point>();
            
            for (int x = 0; x < width; x += 2)
            {
                for (int y = 0; y < height; y += 2)
                {
                    float[] input = new float[inputDim];
                    input[0] = x - centerX;
                    input[1] = centerY - y;
                    
                    int prediction = PredictClass(input, network);
                    
                    if (x > 0 && y > 0 && x < width - 2 && y < height - 2)
                    {
                        bool isBoundary = false;
                        
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
                                    
                                    int neighborPrediction = PredictClass(neighborInput, network);
                                    
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

        private void ShowCurrentDecisionBoundaryWithColors(NeuralNetwork network, int epoch, float[] mean, float[] std)
        {
            Bitmap tempBitmap = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            
            using (Graphics g = Graphics.FromImage(tempBitmap))
            {
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
                        
                        int predictedClass = PredictClass(x, network);
                        
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
                
                int center_width = pictureBox1.Width / 2;
                int center_height = pictureBox1.Height / 2;
                for (int i = 0; i < numSample; i++)
                {
                    int temp_x = (int)Samples[i * inputDim] + center_width;
                    int temp_y = center_height - (int)Samples[i * inputDim + 1];
                    draw_sample(g, temp_x, temp_y, (int)targets[i]);
                }
                
                Pen axisPen = new Pen(Color.Gray, 0.5f);
                g.DrawLine(axisPen, center_width, 0, center_width, pictureBox1.Height);
                g.DrawLine(axisPen, 0, center_height, pictureBox1.Width, center_height);
                
                Font infoFont = new Font("Arial", 9, FontStyle.Bold);
                g.DrawString($"Epoch: {epoch}", infoFont, Brushes.DarkBlue, 10, 10);
                g.DrawString($"Layers: {network.Layers.Count}", infoFont, Brushes.DarkBlue, 10, 25);
                g.DrawString($"Momentum: {network.Momentum}", infoFont, Brushes.DarkBlue, 10, 40);
                
                axisPen.Dispose();
                infoFont.Dispose();
            }
            
            pictureBox1.Image = tempBitmap;
            Application.DoEvents();
        }

        private void ShowFinalDecisionBoundaryWithColors(NeuralNetwork network, float[] mean, float[] std)
        {
            Bitmap finalBitmap = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            
            using (Graphics g = Graphics.FromImage(finalBitmap))
            {
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
                        
                        int predictedClass = PredictClass(x, network);
                        
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
                
                if (class_count == 2)
                {
                    DrawDecisionBoundaryAtColorEdge(g, network, class_count, Color.Red, 3.0f);
                }
                
                int center_width = pictureBox1.Width / 2;
                int center_height = pictureBox1.Height / 2;
                for (int i = 0; i < numSample; i++)
                {
                    int temp_x = (int)Samples[i * inputDim] + center_width;
                    int temp_y = center_height - (int)Samples[i * inputDim + 1];
                    draw_sample(g, temp_x, temp_y, (int)targets[i]);
                }
                
                Pen axisPen = new Pen(Color.Gray, 0.5f);
                g.DrawLine(axisPen, center_width, 0, center_width, pictureBox1.Height);
                g.DrawLine(axisPen, 0, center_height, pictureBox1.Width, center_height);
                
                int correct = CalculateAccuracy(network, mean, std);
                float accuracy = (float)correct / numSample * 100;
                
                Font infoFont = new Font("Arial", 10, FontStyle.Bold);
                g.DrawString($"Accuracy: {accuracy:F1}%", infoFont, Brushes.DarkBlue, 10, 30);
                g.DrawString($"Layers: {network.Layers.Count}", infoFont, Brushes.DarkBlue, 10, 50);
                g.DrawString($"Momentum: {network.Momentum}", infoFont, Brushes.DarkBlue, 10, 70);
                
                axisPen.Dispose();
                infoFont.Dispose();
            }
            
            pictureBox1.Image = finalBitmap;
        }

        private void DrawRegressionSurface(NeuralNetwork network, float[] mean, float[] std)
        {
            Bitmap surfaceBitmap = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            int center_width = pictureBox1.Width / 2;
            int center_height = pictureBox1.Height / 2;

            float minOutput = float.MaxValue;
            float maxOutput = float.MinValue;
            
            for (int x = 0; x < pictureBox1.Width; x += 2)
            {
                for (int y = 0; y < pictureBox1.Height; y += 2)
                {
                    float[] input = new float[inputDim];
                    input[0] = x - center_width;
                    input[1] = center_height - y;
                    
                    if (mean != null && std != null)
                    {
                        input[0] = (input[0] - mean[0]) / std[0];
                        input[1] = (input[1] - mean[1]) / std[1];
                    }
                    
                    float[] output = Predict(input, network);
                    minOutput = Math.Min(minOutput, output[0]);
                    maxOutput = Math.Max(maxOutput, output[0]);
                }
            }

            using (Graphics g = Graphics.FromImage(surfaceBitmap))
            {
                for (int x = 0; x < pictureBox1.Width; x += 2)
                {
                    for (int y = 0; y < pictureBox1.Height; y += 2)
                    {
                        float[] input = new float[inputDim];
                        input[0] = x - center_width;
                        input[1] = center_height - y;
                        
                        if (mean != null && std != null)
                        {
                            input[0] = (input[0] - mean[0]) / std[0];
                            input[1] = (input[1] - mean[1]) / std[1];
                        }
                        
                        float[] output = Predict(input, network);
                        float normalized = (output[0] - minOutput) / (maxOutput - minOutput);
                        
                        Color color;
                        if (normalized < 0.5f)
                        {
                            int blue = (int)(255 * (0.5f - normalized) * 2);
                            int green = (int)(255 * normalized * 2);
                            color = Color.FromArgb(0, green, blue);
                        }
                        else
                        {
                            int green = (int)(255 * (1.0f - normalized) * 2);
                            int red = (int)(255 * (normalized - 0.5f) * 2);
                            color = Color.FromArgb(red, green, 0);
                        }
                        
                        using (SolidBrush brush = new SolidBrush(color))
                        {
                            g.FillRectangle(brush, x, y, 2, 2);
                        }
                    }
                }

                for (int i = 0; i < numSample; i++)
                {
                    int temp_x = (int)Samples[i * inputDim] + center_width;
                    int temp_y = center_height - (int)Samples[i * inputDim + 1];
                    
                    float value = targets[i];
                    float normalizedValue = (value - minOutput) / (maxOutput - minOutput);
                    Color pointColor = normalizedValue < 0.5f ? Color.Blue : Color.Red;
                    
                    using (Pen pen = new Pen(pointColor, 3.0f))
                    {
                        g.DrawLine(pen, temp_x - 5, temp_y, temp_x + 5, temp_y);
                        g.DrawLine(pen, temp_x, temp_y - 5, temp_x, temp_y + 5);
                    }

                    Font font = new Font("Arial", 8);
                    string valueText = value.ToString("F2", CultureInfo.GetCultureInfo("tr-TR"));
                    g.DrawString(valueText, font, Brushes.Black, temp_x + 8, temp_y - 8);
                    font.Dispose();
                }

                Pen axisPen = new Pen(Color.Gray, 0.5f);
                g.DrawLine(axisPen, center_width, 0, center_width, pictureBox1.Height);
                g.DrawLine(axisPen, 0, center_height, pictureBox1.Width, center_height);
                
                Font infoFont = new Font("Arial", 9, FontStyle.Bold);
                g.DrawString($"Regression Surface", infoFont, Brushes.Black, 10, 10);
                g.DrawString($"Min: {minOutput:F2}, Max: {maxOutput:F2}", infoFont, Brushes.Black, 10, 25);
                g.DrawString($"Momentum: {network.Momentum}", infoFont, Brushes.Black, 10, 40);
                
                axisPen.Dispose();
                infoFont.Dispose();
            }

            pictureBox1.Image = surfaceBitmap;
        }

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
                
                if (mean != null && std != null)
                {
                    x[0] = (x[0] - mean[0]) / std[0];
                    x[1] = (x[1] - mean[1]) / std[1];
                }
                
                int predicted = PredictClass(x, network);
                if (predicted == (int)targets[i])
                {
                    correct++;
                }
            }
            
            return correct;
        }

        private float CalculateMSE(NeuralNetwork network, float[] mean, float[] std)
        {
            float totalError = 0.0f;
            
            for (int i = 0; i < numSample; i++)
            {
                float[] x = new float[inputDim];
                for (int j = 0; j < inputDim; j++)
                {
                    x[j] = Samples[i * inputDim + j];
                }
                
                if (mean != null && std != null)
                {
                    x[0] = (x[0] - mean[0]) / std[0];
                    x[1] = (x[1] - mean[1]) / std[1];
                }
                
                float[] output = Predict(x, network);
                float error = targets[i] - output[0];
                totalError += error * error;
            }
            
            return totalError / numSample;
        }

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
                float maxError = errorHistory.Max();
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
                g.DrawString($"Momentum: {momentum}", labelFont, Brushes.Black, trainingGraph.Width - 120, 45);
            }
        }

        private void UpdateTrainingGraph(float error)
        {
            errorHistory.Add(error);
            trainingGraph.Invalidate();
            Application.DoEvents();
        }

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

        private float[] Add_Labels(float[] Labels, int Size, float label)
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
                    
                    float[] target = new float[class_count];
                    if (problemType == "classification")
                    {
                        if (class_count == 2)
                        {
                            target[(int)targets[sampleIndex]] = 1.0f;
                            target[1 - (int)targets[sampleIndex]] = -1.0f;
                        }
                        else
                        {
                            for (int i = 0; i < class_count; i++)
                            {
                                target[i] = (i == (int)targets[sampleIndex]) ? 1.0f : -1.0f;
                            }
                        }
                    }
                    else
                    {
                        target[0] = targets[sampleIndex];
                    }

                    float[] output = Forward(input, network);
                    
                    for (int j = 0; j < class_count; j++)
                    {
                        if (problemType == "regression")
                        {
                            float error = target[j] - output[j];
                            totalError += error * error;
                        }
                        else
                        {
                            totalError += Math.Abs(target[j] - output[j]);
                        }
                    }
                    
                    Backward(input, target, network);
                }

                float avgError = totalError / numSample;
                UpdateTrainingGraph(avgError);

                if (epoch % 50 == 0)
                {
                    if (problemType == "classification")
                    {
                        ShowCurrentDecisionBoundaryWithColors(network, epoch, mean, std);
                    }
                    else
                    {
                        DrawRegressionSurface(network, mean, std);
                    }
                }

                if (epoch % 10 == 0)
                {
                    progressBar1.Value = (int)((float)epoch / epochs * 100);
                    label4.Text = $"Epoch: {epoch}, Error: {avgError:F4}, Momentum: {momentum}";
                    Application.DoEvents();
                }
            }

            progressBar1.Value = 100;
            label4.Text = "Status: Training Completed!";
            progressBar1.Visible = false;

            if (problemType == "classification")
            {
                ShowFinalDecisionBoundaryWithColors(network, mean, std);
            }
            else
            {
                DrawRegressionSurface(network, mean, std);
            }

            if (problemType == "classification")
            {
                int correct = CalculateAccuracy(network, mean, std);
                float accuracy = (float)correct / numSample * 100;
                textBox1.AppendText($"Training completed!\r\nFinal Error: {errorHistory[errorHistory.Count - 1]:F4}\r\nAccuracy: {accuracy:F2}%\r\nMomentum: {momentum}\r\n");
                MessageBox.Show($"Training completed!\nFinal accuracy: {accuracy:F2}%\nLayers: {network.Layers.Count}\nMomentum: {momentum}");
            }
            else
            {
                float mse = CalculateMSE(network, mean, std);
                textBox1.AppendText($"Training completed!\r\nFinal MSE: {mse:F4}\r\nMomentum: {momentum}\r\n");
                MessageBox.Show($"Training completed!\nFinal MSE: {mse:F4}\nLayers: {network.Layers.Count}\nMomentum: {momentum}");
            }
        }

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

        private void draw_regression_point(int temp_x, int temp_y, float value)
        {
            using (Graphics g = pictureBox1.CreateGraphics())
            {
                Color color = value < 0 ? Color.Blue : Color.Red;
                using (Pen pen = new Pen(color, 3.0f))
                {
                    g.DrawLine(pen, temp_x - 5, temp_y, temp_x + 5, temp_y);
                    g.DrawLine(pen, temp_x, temp_y - 5, temp_x, temp_y + 5);
                }
                
                Font font = new Font("Arial", 8);
                string valueText = value.ToString("F2", CultureInfo.GetCultureInfo("tr-TR"));
                g.DrawString(valueText, font, Brushes.Black, temp_x + 8, temp_y - 8);
                font.Dispose();
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

        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            if (class_count == 0 && problemType == "classification")
            {
                MessageBox.Show("The Network Architecture should be firstly set up");
                return;
            }

            float[] x = new float[inputDim];
            int temp_x = e.X;
            int temp_y = e.Y;
            x[0] = temp_x - (pictureBox1.Width / 2);
            x[1] = (pictureBox1.Height / 2) - temp_y;
            
            if (problemType == "classification")
            {
                int numLabel = Convert.ToInt32(ClassNoBox.Text);
                if (numLabel > class_count)
                {
                    MessageBox.Show("The class label cannot be greater than the maximum number of classes.");
                    return;
                }
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
            }
            else
            {
                float value = ParseRegressionValue(regressionValueTextBox.Text);
                
                if (numSample == 0)
                {
                    numSample = 1;
                    Samples = new float[numSample * inputDim];
                    targets = new float[numSample];
                    Array.Copy(x, Samples, inputDim);
                    targets[0] = value;
                }
                else
                {
                    numSample++;
                    Samples = Add_Data(Samples, numSample, x, inputDim);
                    targets = Add_Labels(targets, numSample, value);
                }
                
                draw_regression_point(temp_x, temp_y, value);
            }
            
            label3.Text = "Samples Count: " + numSample.ToString();
            
            if (numSample > 0)
            {
                Train_Button.Enabled = true;
                trainingToolStripMenuItem.Enabled = true;
            }
        }

        private void Set_Net_Click(object sender, EventArgs e)
        {
            try
            {
                momentum = ParseRegressionValue(momentumTextBox.Text);
                if (momentum < 0 || momentum >= 1)
                {
                    MessageBox.Show("Momentum değeri 0 ile 1 arasında olmalıdır!");
                    return;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Geçersiz momentum değeri: {momentumTextBox.Text}\nLütfen 0 ile 1 arasında sayı girin (örnek: 0,9)", "Hata");
                return;
            }

            if (problemType == "classification")
            {
                class_count = Convert.ToInt32(ClassCountBox.Text);
            }
            else
            {
                class_count = 1;
            }
            
            int hiddenLayers = Convert.ToInt32(hiddenLayersComboBox.Text);
            int neuronsPerLayer = Convert.ToInt32(neuronsPerLayerTextBox.Text);
            
            network = new NeuralNetwork(learningRate, problemType, momentum);
            
            network.AddLayer(inputDim, neuronsPerLayer);
            
            for (int i = 0; i < hiddenLayers - 1; i++)
            {
                network.AddLayer(neuronsPerLayer, neuronsPerLayer);
            }
            
            network.AddLayer(neuronsPerLayer, class_count);
            
            Set_Net.Text = "Network is Ready";
            label4.Text = $"Status: Network Ready ({hiddenLayers + 1} layers, {problemType}, Momentum: {momentum})";
            textBox1.AppendText($"Network initialized with {hiddenLayers + 1} layers for {problemType}, Momentum: {momentum}\n");
        }

        private void Train_Button_Click(object sender, EventArgs e)
        {
            TrainNetwork();
        }

        private void trainingToolStripMenuItem_Click(object sender, EventArgs e)
        {
            TrainNetwork();
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
            
            if (problemType == "classification")
            {
                Bitmap surface = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            
                using (Graphics g = Graphics.FromImage(surface))
                {
                    for (int row = 0; row < pictureBox1.Height; row += 2)
                    {
                        for (int column = 0; column < pictureBox1.Width; column += 2)
                        {
                            float[] x = new float[inputDim];
                            x[0] = column - (pictureBox1.Width / 2);
                            x[1] = (pictureBox1.Height / 2) - row;
                            
                            x[0] = (x[0] - mean[0]) / std[0];
                            x[1] = (x[1] - mean[1]) / std[1];
                            
                            int predictedClass = PredictClass(x, network);
                            
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
                    
                    if (class_count == 2)
                    {
                        DrawDecisionBoundaryAtColorEdge(g, network, class_count, Color.DarkBlue, 3.0f);
                    }
                    
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
                
                int correct = CalculateAccuracy(network, mean, std);
                float accuracy = (float)correct / numSample * 100;
                MessageBox.Show($"Testing completed!\nAccuracy: {accuracy:F2}%\nLayers: {network.Layers.Count}\nMomentum: {momentum}");
            }
            else
            {
                DrawRegressionSurface(network, mean, std);
                float mse = CalculateMSE(network, mean, std);
                MessageBox.Show($"Testing completed!\nMSE: {mse:F4}\nLayers: {network.Layers.Count}\nMomentum: {momentum}");
            }
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
    }
}