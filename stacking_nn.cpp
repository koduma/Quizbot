//g++ -O2 stacking_nn.cpp -o snn

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

struct Sample
{
    double target;
    double feature[5];
};

class StackingNN
{
public:

    static constexpr int INPUT = 5;
    static constexpr int HIDDEN = 16;

    double W1[INPUT][HIDDEN];
    double b1[HIDDEN];

    double W2[HIDDEN];
    double b2;

    double lr;

    StackingNN(double learningRate=0.001) // 収束を早めるため学習率を微調整
        : lr(learningRate)
    {
        initialize();
    }

    void initialize()
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::normal_distribution<double> d1(0.0, std::sqrt(2.0/INPUT));
        std::normal_distribution<double> d2(0.0, std::sqrt(2.0/HIDDEN));

        for(int i=0;i<INPUT;i++) {
            for(int j=0;j<HIDDEN;j++) {
                W1[i][j]=d1(gen);
            }
        }

        for(int j=0;j<HIDDEN;j++) {
            W2[j]=d2(gen);
            b1[j]=0.0;
        }
        b2=0.0;
    }

    static double relu(double x) { return x > 0.0 ? x : 0.0; }

    static double sigmoid(double x)
    {
        if(x > 30.0) return 1.0;
        if(x < -30.0) return 0.0;
        return 1.0/(1.0+std::exp(-x));
    }

    double predict(const double x[INPUT]) const
    {
        double hidden[HIDDEN];
        for(int j=0;j<HIDDEN;j++) {
            double z=b1[j];
            for(int i=0;i<INPUT;i++) {
                z+=x[i]*W1[i][j];
            }
            hidden[j]=relu(z);
        }
        double y=b2;
        for(int j=0;j<HIDDEN;j++) {
            y+=hidden[j]*W2[j];
        }
        return sigmoid(y);
    }

    // クラス不均衡対策として、重み(sampleWeight)を乗算できるように拡張
    double train(const double x[INPUT], double target, double sampleWeight = 1.0)
    {
        double hiddenRaw[HIDDEN];
        double hidden[HIDDEN];

        // Forward
        for(int j=0;j<HIDDEN;j++) {
            double z=b1[j];
            for(int i=0;i<INPUT;i++) {
                z+=x[i]*W1[i][j];
            }
            hiddenRaw[j]=z;
            hidden[j]=relu(z);
        }

        double y=b2;
        for(int j=0;j<HIDDEN;j++) {
            y+=hidden[j]*W2[j];
        }
        double p=sigmoid(y);

        // BCE Loss
        constexpr double eps=1e-12;
        double loss= -sampleWeight * (target*std::log(p+eps) + (1.0-target)*std::log(1.0-p+eps));

        // Backprop
        double dy= sampleWeight * (p-target);

        double oldW2[HIDDEN];
        for(int j=0;j<HIDDEN;j++) {
            oldW2[j]=W2[j];
        }

        // W2
        for(int j=0;j<HIDDEN;j++) {
            W2[j]-= lr*dy*hidden[j];
        }
        b2-=lr*dy;

        // W1
        for(int j=0;j<HIDDEN;j++) {
            double dz= dy*oldW2[j];
            if(hiddenRaw[j]<=0.0) {
                dz=0.0;
            }
            for(int i=0;i<INPUT;i++) {
                W1[i][j]-= lr*dz*x[i];
            }
            b1[j]-= lr*dz;
        }

        return loss;
    }

    void saveWeights(const std::string& filename) const
    {
        std::ofstream out(filename);
        for(int i=0; i<INPUT; i++) {
            for(int j=0; j<HIDDEN; j++) {
                out << W1[i][j] << (j==HIDDEN-1 ? "" : " ");
            }
            out << "\n";
        }
        for(int j=0; j<HIDDEN; j++) {
            out << b1[j] << (j==HIDDEN-1 ? "" : " ");
        }
        out << "\n";
        for(int j=0; j<HIDDEN; j++) {
            out << W2[j] << (j==HIDDEN-1 ? "" : " ");
        }
        out << "\n";
        out << b2 << "\n";
    }
};

int main()
{
    std::cout << "Loading CSV data..." << std::endl;
    std::vector<Sample> samples;
    std::ifstream file("stacking_training_data.csv");
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open stacking_training_data.csv" << std::endl;
        return 1;
    }

    std::string line;
    std::getline(file, line); // ヘッダーをスキップ

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        
        // 0: is_correct
        std::getline(ss, token, ',');
        double is_correct = std::stod(token);
        
        // 1: candidate_id (読み飛ばす)
        std::getline(ss, token, ',');
        
        Sample s;
        s.target = is_correct;
        
        // 2~6: features (BoW, Jaccard, Order, BM25, Cross)
        for (int i = 0; i < 5; ++i) {
            std::getline(ss, token, ',');
            s.feature[i] = std::stod(token);
        }
        samples.push_back(s);
    }

    std::cout << "Loaded " << samples.size() << " samples." << std::endl;

    StackingNN net(0.001);
    const int EPOCHS = 1000;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "Training started..." << std::endl;
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        double totalLoss = 0.0;

        // エポックごとにシャッフル（SGDの基本）
        std::shuffle(samples.begin(), samples.end(), gen);

        for(const auto& sample : samples) {
            // クラス不均衡対策：1個の正解と15個のハズレがあるため、正解時の重みを15倍にして学習
            double weight = (sample.target == 1.0) ? 15.0 : 1.0;
            totalLoss += net.train(sample.feature, sample.target, weight);
        }

        if(epoch == 0 || (epoch + 1) % 10 == 0) {
            std::cout << "Epoch [" << (epoch+1) << "/" << EPOCHS 
                      << "] Loss: " << (totalLoss / samples.size()) << std::endl;
        }
    }

    net.saveWeights("nn_weights.txt");
    std::cout << "Training complete! Weights saved to 'nn_weights.txt'." << std::endl;

    return 0;
}