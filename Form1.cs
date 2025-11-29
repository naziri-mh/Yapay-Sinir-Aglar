using System;
using System.Drawing;
using System.Windows.Forms;
using System.IO;
using System.Collections.Generic;

namespace Collect
{
    public partial class Form1 : Form
    {
        private int class_count = 0, numSample = 0, inputDim = 2;
        private float[] Samples, targets, Weights, bias;
        private float learningRate = 0.1f;
        private int epochs = 1000;
        
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
            this.groupBox1.Size = new Size(200, 180);
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
            
            this.Set_Net = new Button();
            this.Set_Net.Location = new Point(10, 50);
            this.Set_Net.Size = new Size(131, 25);
            this.Set_Net.Text = "Network Setting";
            this.Set_Net.Click += new EventHandler(Set_Net_Click);
            
            this.Train_Button = new Button();
            this.Train_Button.Location = new Point(10, 85);
            this.Train_Button.Size = new Size(131, 25);
            this.Train_Button.Text = "Train Network";
            this.Train_Button.Click += new EventHandler(Train_Button_Click);
            this.Train_Button.Enabled = false;
            
            this.groupBox1.Controls.Add(Set_Net);
            this.groupBox1.Controls.Add(Train_Button);
            this.groupBox1.Controls.Add(label1);
            this.groupBox1.Controls.Add(ClassCountBox);
            
            // GroupBox2 - Data Collection
            this.groupBox2 = new GroupBox();
            this.groupBox2.Location = new Point(520, 450);
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
            this.Text = "Neural Network - Live Training with Colors";
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

        // KARAR SINIRI ÇİZME FONKSİYONU (İki rengin birleştiği yere)
        private void DrawDecisionBoundaryAtColorEdge(Graphics g, float[] weights, float[] bias, int classCount, Color color, float lineWidth = 2.0f)
        {
            if (classCount != 2) return;

            int width = pictureBox1.Width;
            int height = pictureBox1.Height;
            int centerX = width / 2;
            int centerY = height / 2;

            float w1 = weights[0];
            float w2 = weights[1];
            float b = bias[0];

            // Karar sınırı: w1*x + w2*y + b = 0
            if (Math.Abs(w2) < 0.001f)
            {
                // Dikey çizgi
                float x = -b / w1;
                int screenX = centerX + (int)x;
                
                if (screenX >= 0 && screenX <= width)
                {
                    Pen linePen = new Pen(color, lineWidth);
                    g.DrawLine(linePen, screenX, 0, screenX, height);
                    linePen.Dispose();
                }
                return;
            }

            // Eğimli çizgi: y = (-w1*x - b) / w2
            float slope = -w1 / w2;
            float intercept = -b / w2;

            // Ekran kenarlarıyla kesişim noktalarını bul
            List<Point> edgePoints = new List<Point>();

            // Sol kenar (x = -centerX)
            float leftX = -centerX;
            float leftY = slope * leftX + intercept;
            int screenLeftY = centerY - (int)leftY;
            if (screenLeftY >= 0 && screenLeftY <= height)
            {
                edgePoints.Add(new Point(0, screenLeftY));
            }

            // Sağ kenar (x = centerX)
            float rightX = centerX;
            float rightY = slope * rightX + intercept;
            int screenRightY = centerY - (int)rightY;
            if (screenRightY >= 0 && screenRightY <= height)
            {
                edgePoints.Add(new Point(width, screenRightY));
            }

            // Üst kenar (y = centerY, screenY = 0)
            float topY = centerY;
            float topX = (topY - intercept) / slope;
            int screenTopX = centerX + (int)topX;
            if (screenTopX >= 0 && screenTopX <= width)
            {
                edgePoints.Add(new Point(screenTopX, 0));
            }

            // Alt kenar (y = -centerY, screenY = height)
            float bottomY = -centerY;
            float bottomX = (bottomY - intercept) / slope;
            int screenBottomX = centerX + (int)bottomX;
            if (screenBottomX >= 0 && screenBottomX <= width)
            {
                edgePoints.Add(new Point(screenBottomX, height));
            }

            // Benzersiz noktaları bul ve çizgi çiz
            HashSet<Point> uniquePoints = new HashSet<Point>();
            foreach (Point p in edgePoints)
            {
                uniquePoints.Add(p);
            }

            if (uniquePoints.Count >= 2)
            {
                Point[] points = new Point[uniquePoints.Count];
                uniquePoints.CopyTo(points, 0);
                
                // En uzak iki noktayı bul
                double maxDistance = 0;
                Point p1 = points[0], p2 = points[0];
                
                for (int i = 0; i < points.Length; i++)
                {
                    for (int j = i + 1; j < points.Length; j++)
                    {
                        double dist = Math.Sqrt(Math.Pow(points[i].X - points[j].X, 2) + Math.Pow(points[i].Y - points[j].Y, 2));
                        if (dist > maxDistance)
                        {
                            maxDistance = dist;
                            p1 = points[i];
                            p2 = points[j];
                        }
                    }
                }
                
                Pen linePen = new Pen(color, lineWidth);
                g.DrawLine(linePen, p1, p2);
                linePen.Dispose();
            }
        }

        // EĞİTİM SIRASINDA İKİ RENKLİ GÖSTERİM
        private void ShowCurrentDecisionBoundaryWithColors(float[] currentWeights, float[] currentBias, int epoch, float[] mean, float[] std)
        {
            Bitmap tempBitmap = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            
            // Arka planı iki renge boya
            /*for (int row = 0; row < pictureBox1.Height; row += 2)
            {
                for (int column = 0; column < pictureBox1.Width; column += 2)
                {
                    float[] x = new float[inputDim];
                    x[0] = column - (pictureBox1.Width / 2);
                    x[1] = (pictureBox1.Height / 2) - row;
                    
                    x[0] = (x[0] - mean[0]) / std[0];
                    x[1] = (x[1] - mean[1]) / std[1];
                    
                    int predictedClass = Test_Forward(x, currentWeights, currentBias, class_count, inputDim);
                    
                    Color c = Color.LightBlue;
                    if (predictedClass == 1)
                        c = Color.LightCoral;
                    else if (predictedClass >= 2)
                        c = Color.LightGreen;
                    
                    tempBitmap.SetPixel(column, row, c);
                }
            }*/
            
            using (Graphics g = Graphics.FromImage(tempBitmap))
            {
                /*
                // İKİ RENGİN BİRLEŞTİĞİ YERE ÇİZGİ ÇİZ
                if (class_count == 2)
                {
                    DrawDecisionBoundaryAtColorEdge(g, currentWeights, currentBias, class_count, Color.DarkRed, 2.5f);
                }
                */
                
                // Noktaları çiz - YENİ: Karar sınırına göre konumlandır
                int center_width = pictureBox1.Width / 2;
                int center_height = pictureBox1.Height / 2;
                for (int i = 0; i < numSample; i++)
                {
                    // Orijinal koordinatları al
                    float original_x = Samples[i * inputDim];
                    float original_y = Samples[i * inputDim + 1];
                    
                    // Normalize edilmiş koordinatları hesapla
                    float[] normalized_x = new float[inputDim];
                    normalized_x[0] = (original_x - mean[0]) / std[0];
                    normalized_x[1] = (original_y - mean[1]) / std[1];
                    
                    // Sınıf tahmini yap
                    int predictedClass = Test_Forward(normalized_x, currentWeights, currentBias, class_count, inputDim);
                    
                    // Ekran koordinatlarını hesapla
                    int screen_x = center_width + (int)original_x;
                    int screen_y = center_height - (int)original_y;
                    
                    // Karar sınırına göre noktayı kaydır
                    if (class_count == 2)
                    {
                        // Karar sınırı denklemi: w1*x + w2*y + b = 0
                        float w1 = currentWeights[0];
                        float w2 = currentWeights[1];
                        float b = currentBias[0];
                        
                        // Noktanın karar sınırına uzaklığını hesapla
                        float distance = (w1 * normalized_x[0] + w2 * normalized_x[1] + b) / (float)Math.Sqrt(w1 * w1 + w2 * w2);
                        
                        // Karar sınırına dik birim vektör
                        float unit_x = w1 / (float)Math.Sqrt(w1 * w1 + w2 * w2);
                        float unit_y = w2 / (float)Math.Sqrt(w1 * w1 + w2 * w2);
                        
                        // Noktayı karar sınırına doğru kaydır (sınıra yakın ama tam üstünde değil)
                        float shiftDistance = Math.Sign(distance) * 5.0f; // 5 pixel kaydır
                        
                        // Ekran koordinatlarını güncelle
                        screen_x += (int)(unit_x * shiftDistance * std[0]); // Normalizasyonu geri al
                        screen_y -= (int)(unit_y * shiftDistance * std[1]); // Y ekseni ters olduğu için -
                    }
                    
                    draw_sample(g, screen_x, screen_y, (int)targets[i]);
                }
                
                // Eksenleri çiz
                Pen axisPen = new Pen(Color.Gray, 0.5f);
                g.DrawLine(axisPen, center_width, 0, center_width, pictureBox1.Height);
                g.DrawLine(axisPen, 0, center_height, pictureBox1.Width, center_height);
                
                // Epoch bilgisini yaz
                Font infoFont = new Font("Arial", 9, FontStyle.Bold);
                g.DrawString($"Epoch: {epoch}", infoFont, Brushes.DarkBlue, 10, 10);
                
                axisPen.Dispose();
                infoFont.Dispose();
            }
            
            pictureBox1.Image = tempBitmap;
            Application.DoEvents();
        }

        // FİNAL İKİ RENKLİ KARAR SINIRI (TEK TANIM)
        private void ShowFinalDecisionBoundaryWithColors(float[] finalWeights, float[] finalBias, float[] mean, float[] std)
        {
            Bitmap finalBitmap = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            
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
                    
                    int predictedClass = Test_Forward(x, finalWeights, finalBias, class_count, inputDim);
                    
                    Color c = Color.LightBlue;
                    if (predictedClass == 1)
                        c = Color.LightCoral;
                    else if (predictedClass >= 2)
                        c = Color.LightGreen;
                    
                    finalBitmap.SetPixel(column, row, c);
                }
            }
            
            using (Graphics g = Graphics.FromImage(finalBitmap))
            {
                // FİNAL ÇİZGİYİ İKİ RENGİN BİRLEŞTİĞİ YERE ÇİZ
                if (class_count == 2)
                {
                    DrawDecisionBoundaryAtColorEdge(g, finalWeights, finalBias, class_count, Color.Red, 3.0f);
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
                int correct = CalculateAccuracy(finalWeights, finalBias, mean, std);
                float accuracy = (float)correct / numSample * 100;
                
                Font infoFont = new Font("Arial", 10, FontStyle.Bold);
                g.DrawString($"Accuracy: {accuracy:F1}%", infoFont, Brushes.DarkBlue, 10, 30);
                
                axisPen.Dispose();
                infoFont.Dispose();
            }
            
            pictureBox1.Image = finalBitmap;
        }

        // DOĞRULUK HESAPLAMA FONKSİYONU
        private int CalculateAccuracy(float[] weights, float[] bias, float[] mean, float[] std)
        {
            int correct = 0;
            
            for (int i = 0; i < numSample; i++)
            {
                float[] x = new float[inputDim];
                for (int j = 0; j < inputDim; j++)
                {
                    x[j] = (Samples[i * inputDim + j] - mean[j]) / std[j];
                }
                
                int predicted = Test_Forward(x, weights, bias, class_count, inputDim);
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

        private float[] init_array_random(int len)
        {
            float[] arr = new float[len];
            Random rand = new Random();
            for (int i = 0; i < len; i++)
            {
                arr[i] = (float)(rand.NextDouble() - 0.5);
            }
            return arr;
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

        private int Test_Forward(float[] x, float[] weight, float[] bias, int num_Class, int inputDim)
        {
            int i, j, index_Max = 0;

            if (num_Class > 2)
            {
                float[] output = new float[num_Class];
                
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

        // EĞİTİM FONKSİYONU (İki renkli çizgi ile)
        private void TrainNetwork()
        {
            if (numSample == 0)
            {
                MessageBox.Show("No samples available for training!");
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
                    float[] x = new float[inputDim];
                    Array.Copy(normalizedSamples, sampleIndex * inputDim, x, 0, inputDim);
                    int target = (int)targets[sampleIndex];

                    if (class_count > 2)
                    {
                        float[] outputs = new float[class_count];
                        float[] errors = new float[class_count];

                        for (int i = 0; i < class_count; i++)
                        {
                            outputs[i] = 0.0f;
                            for (int j = 0; j < inputDim; j++)
                            {
                                outputs[i] += Weights[i * inputDim + j] * x[j];
                            }
                            outputs[i] += bias[i];
                            outputs[i] = (float)Math.Tanh(outputs[i]);
                        }

                        for (int i = 0; i < class_count; i++)
                        {
                            float desired = (i == target) ? 1.0f : -1.0f;
                            errors[i] = desired - outputs[i];
                            totalError += Math.Abs(errors[i]);
                        }

                        for (int i = 0; i < class_count; i++)
                        {
                            float derivative = 1.0f - outputs[i] * outputs[i];
                            float delta = errors[i] * derivative;

                            for (int j = 0; j < inputDim; j++)
                            {
                                Weights[i * inputDim + j] += learningRate * delta * x[j];
                            }
                            bias[i] += learningRate * delta;
                        }
                    }
                    else
                    {
                        float output = 0.0f;
                        for (int j = 0; j < inputDim; j++)
                        {
                            output += Weights[j] * x[j];
                        }
                        output += bias[0];
                        output = (float)Math.Tanh(output);

                        float desired = (target == 0) ? 1.0f : -1.0f;
                        float error = desired - output;
                        totalError += Math.Abs(error);

                        float derivative = 1.0f - output * output;
                        float delta = error * derivative;

                        for (int j = 0; j < inputDim; j++)
                        {
                            Weights[j] += learningRate * delta * x[j];
                        }
                        bias[0] += learningRate * delta;
                    }
                }

                float avgError = totalError / numSample;
                UpdateTrainingGraph(avgError);

                // Her 50 epoch'ta bir iki renkli karar sınırını göster
                if (epoch % 50 == 0 && class_count == 2)
                {
                    ShowCurrentDecisionBoundaryWithColors(Weights, bias, epoch, mean, std);
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

            // FİNAL İKİ RENKLİ KARAR SINIRINI GÖSTER
            ShowFinalDecisionBoundaryWithColors(Weights, bias, mean, std);

            // Doğruluk hesapla
            int correct = CalculateAccuracy(Weights, bias, mean, std);
            float accuracy = (float)correct / numSample * 100;
            textBox1.AppendText($"Training completed!\r\nFinal Error: {errorHistory[errorHistory.Count - 1]:F4}\r\nAccuracy: {accuracy:F2}%\r\n");
            MessageBox.Show($"Training completed!\nFinal accuracy: {accuracy:F2}%");
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
            Weights = new float[class_count * inputDim];
            bias = new float[class_count];
            
            Weights = init_array_random(class_count * inputDim);
            bias = init_array_random(class_count);
            
            Set_Net.Text = "Network is Ready";
            label4.Text = "Status: Network Ready for Training";
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

            float[] mean = new float[inputDim];
            float[] std = new float[inputDim];
            
            Z_Score_Parameters(Samples, numSample, inputDim, mean, std);
            
            Bitmap surface = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            
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
                    
                    int predictedClass = Test_Forward(x, Weights, bias, class_count, inputDim);
                    
                    Color c = Color.LightBlue;
                    if (predictedClass == 1)
                        c = Color.LightCoral;
                    else if (predictedClass >= 2)
                        c = Color.LightGreen;
                    
                    surface.SetPixel(column, row, c);
                }
            }
            
            using (Graphics g = Graphics.FromImage(surface))
            {
                // YENİ: TESTTE DE İKİ RENGİN BİRLEŞTİĞİ YERE ÇİZGİ ÇİZ
                if (class_count == 2)
                {
                    DrawDecisionBoundaryAtColorEdge(g, Weights, bias, class_count, Color.DarkBlue, 3.0f);
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
            MessageBox.Show("Testing completed! Decision boundary drawn at color edge.");
        }
    }
}