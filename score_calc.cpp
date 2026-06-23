//Linux:g++ -O3 -shared -fPIC score_calc.cpp -o score_calc.so
//Windows:g++ -O3 -shared -static -static-libgcc -static-libstdc++ score_calc.cpp -o score_calc.dll

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_map>

std::unordered_map<uint64_t, int> delta_map;

extern "C" {
    // 強化学習の動的重み(delta_weights)をC++に同期する
    void init_deltas(int size, const uint64_t* keys, const int* weights) {
        delta_map.clear();
        for(int i = 0; i < size; i++) {
            delta_map[keys[i]] += weights[i];
        }
    }

    // Levenshtein距離
    int levenshtein(const char *s1, const char *s2) {
        int len1 = strlen(s1), len2 = strlen(s2);
        std::vector<int> col(len2 + 1), prevCol(len2 + 1);
        for (int i = 0; i <= len2; i++) prevCol[i] = i;
        for (int i = 0; i < len1; i++) {
            col[0] = i + 1;
            for (int j = 0; j < len2; j++) {
                int cost = (toupper(s1[i]) == toupper(s2[j])) ? 0 : 1;
                col[j + 1] = std::min({prevCol[j + 1] + 1, col[j] + 1, prevCol[j] + cost});
            }
            prevCol = col;
        }
        return prevCol[len2];
    }

    std::string to_upper(const char *s) {
        std::string res = s;
        for(auto &c : res) c = toupper(c);
        return res;
    }

    std::string to_lower(const char *s) {
        std::string res = s;
        for(auto &c : res) c = tolower(c);
        return res;
    }

    bool is_same_word(const char* s1, const char* s2) {
        std::string S1 = to_lower(s1), S2 = to_lower(s2);
        if (S1 == S2) return true;
        if (abs((int)S1.length() - (int)S2.length()) > 1) return false;
        if (std::min(S1.length(), S2.length()) <= 3) return false;
        if (S1[0] != S2[0]) return false;
        return levenshtein(S1.c_str(), S2.c_str()) <= 1;
    }

    bool is_english_word(const char *s) {
        int len = strlen(s);
        if(len == 0) return false;
        for(int i = 0; i < len; i++) {
            if(!isalpha(s[i])) return false;
        }
        return true;
    }

    bool is_capitalized(const char *s) {
        if (strlen(s) == 0) return false;
        if (!isupper(s[0])) return false;
        for(int i = 1; i < strlen(s); i++) {
            if (isupper(s[i])) return false;
        }
        return true;
    }

    bool is_include_cpp(int w_st, int w_ts, const char *s, const char *t) {
        if(w_st == 0 || w_ts == 0) return false;
        if(w_st > 5 || w_ts > 5) {
            std::string S = to_upper(s), T = to_upper(t);
            if(S.length() >= T.length()) return S.find(T) != std::string::npos;
            else return T.find(S) != std::string::npos;
        }
        return false;
    }

    uint16_t get_raw_weight(int p_id, int c_id, const uint32_t* offsets, const uint32_t* indices, const uint16_t* w_data) {
        if (p_id < 0 || c_id < 0) return 0;
        uint32_t start = offsets[p_id], end = offsets[p_id + 1];
        uint16_t val = 0;
        if (start < end) {
            const uint32_t* it = std::lower_bound(indices + start, indices + end, (uint32_t)c_id);
            if (it != indices + end && *it == c_id) val = w_data[std::distance(indices, it)];
        }
        uint64_t key = ((uint64_t)p_id << 32) | (uint32_t)c_id;
        if (delta_map.count(key)) val += delta_map[key];
        return val;
    }

    double get_weight_fast_cpp(int p_id, int c_id, int child_noans, const uint32_t* offsets, const uint32_t* indices, const uint16_t* w_data, int taboo) {
        uint16_t raw = get_raw_weight(p_id, c_id, offsets, indices, w_data);
        if (raw == 0) return 0.0;
        double freq = (child_noans == -1) ? 1.0 : std::max(1.0, (double)child_noans);
        double need = log10((double)taboo / freq) + 1.0;
        return (double)raw * std::max(1.0, need);
    }

    // メインの2重ループ（Pythonコードの完全な翻訳）
    void run_quiz_loop(
        int cand_size, const int* cand_ids, const char** cand_strs, const char** syn_strs,
        const double* uniq_vals, const double* wq_hints, const int* ngram_flags, const int* new_word_flags,
        int quiz_size, const int* quiz_ids, const char** quiz_strs, const int* quiz_noans,
        const uint32_t* offsets, const uint32_t* indices, const uint16_t* w_data,
        int taboo, int maxhit, int is_select_mode,
        double* out_scores, int* out_include, int* out_go_syn
    ) {
        for (int i = 0; i < cand_size; i++) {
            int cand_id = cand_ids[i];
            const char* target_str = cand_strs[i];
            const char* syn_str = syn_strs[i];
            double sum = 1.0;

            if (wq_hints[i] != 0.0) {
                sum = (double)pow(2, maxhit - 1);
                sum *= uniq_vals[i];
            } else {
                sum = uniq_vals[i];
            }

            if (ngram_flags[i] == 1) {
                sum = 1.0;
                if (!is_select_mode) { out_scores[i] = sum; continue; }
            }

            if (new_word_flags[i] == 1) {
                sum = 1.0;
                if (!is_select_mode) { out_scores[i] = sum; continue; }
            }

            bool found_syn = (strlen(syn_str) > 0);
            bool go_syn = false;
            std::string target_upper = to_upper(target_str);
            std::string syn_upper = to_upper(syn_str);

            int cnt = -1;

            for (int j = 0; j < quiz_size; j++) {
                const char* xxx = quiz_strs[j];
                std::string xxx_lower = to_lower(xxx); 

                bool in_noans = (quiz_noans[j] != -1);
                if (in_noans) {
                    if (quiz_noans[j] > taboo && xxx_lower != "water" && xxx_lower != "1") continue;
                } else {
                    continue; // if str(xxx) not in NoAns: continue (Pythonのコード準拠)
                }

                cnt += 1;
                if (strcmp(xxx, "?") == 0 || strcmp(xxx, "!") == 0) continue;

                std::string xxx_upper = to_upper(xxx);

                if (is_same_word(xxx, target_str)) {
                    sum = 1.0;
                    break;
                }

                int dist = levenshtein(target_upper.c_str(), xxx_upper.c_str());
                if (dist < 1) {
                    if (!found_syn) {
                        sum = 1.0;
                        break;
                    }
                    go_syn = true;
                }

                if (found_syn) {
                    int dist2 = levenshtein(syn_upper.c_str(), xxx_upper.c_str());
                    if (dist2 < 1) {
                        go_syn = false;
                        sum *= 100.0;
                    }
                }

                bool bbb = false;
                int w_st = get_raw_weight(cand_id, quiz_ids[j], offsets, indices, w_data);
                int w_ts = get_raw_weight(quiz_ids[j], cand_id, offsets, indices, w_data);
                
                if (is_include_cpp(w_st, w_ts, target_str, xxx)) {
                    for (int w1 = 0; w1 < quiz_size; w1++) {
                        if (bbb) break;
                        for (int w2 = w1 + 1; w2 < quiz_size; w2++) {
                            std::string w3 = std::string(quiz_strs[w1]) + std::string(quiz_strs[w2]);
                            std::string w4 = std::string(quiz_strs[w2]) + std::string(quiz_strs[w1]);
                            
                            int d1 = levenshtein(target_upper.c_str(), to_upper(w3.c_str()).c_str());
                            int d2 = levenshtein(target_upper.c_str(), to_upper(w4.c_str()).c_str());
                            
                            if (d1 <= 1 || d2 <= 1) {
                                out_include[i] = 1;
                                bbb = true;
                            }
                        }
                    }
                }

                double wqz = get_weight_fast_cpp(cand_id, quiz_ids[j], quiz_noans[j], offsets, indices, w_data, taboo);
                
                if (wqz == 0.0) {
                    sum /= 1.2;
                } 
                if (wqz != 0.0) {
                    double weight = (cnt < 5) ? 3.0 : 1.0;
                    sum *= weight * wqz;

                    if (quiz_noans[j] > taboo) {
                        if (xxx_lower != "water" && xxx_lower != "1") {
                            sum /= (weight * wqz);
                        }
                    }

                    if (quiz_noans[j] <= taboo && is_english_word(xxx) && is_capitalized(xxx)) {
                        sum *= 3.0;
                    }
                }
            }

            out_scores[i] = sum;
            out_go_syn[i] = go_syn ? 1 : 0;
        }
    }
}
