// apps/llm_cli.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "llama.h"

// ---------- tiny arg parser ----------
static const char *get_arg(int &i, int argc, char **argv)
{
    if (i + 1 >= argc)
        return nullptr;
    return argv[++i];
}

// ---------- UTF-8 console ----------
static void win32_enable_utf8_console()
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

// ---------- stop sequence detector ----------
static bool ends_with_any(const std::string &s, const std::vector<std::string> &stops)
{
    for (const auto &t : stops)
    {
        if (t.empty())
            continue;
        if (s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0)
            return true;
    }
    return false;
}

static void trim_at_stop_first_occurrence(std::string &s, const std::vector<std::string> &stops)
{
    size_t cut = std::string::npos;
    for (const auto &t : stops)
    {
        if (t.empty())
            continue;
        size_t pos = s.find(t);
        if (pos != std::string::npos)
            cut = std::min(cut, pos);
    }
    if (cut != std::string::npos)
        s.resize(cut);
}

// ---------- force "one sentence" postprocess ----------
static size_t find_first_sentence_end_zh(const std::string &s)
{
    // UTF-8 for "。！？"
    static const std::vector<std::string> ends = {u8"。", u8"！", u8"？"};
    size_t best = std::string::npos;
    for (const auto &e : ends)
    {
        size_t p = s.find(e);
        if (p != std::string::npos)
        {
            size_t endpos = p + e.size();
            best = (best == std::string::npos) ? endpos : std::min(best, endpos);
        }
    }
    // also treat newline as sentence end if it appears earlier
    size_t nl = s.find('\n');
    if (nl != std::string::npos)
    {
        best = (best == std::string::npos) ? nl : std::min(best, nl);
    }
    return best;
}

static void normalize_one_sentence(std::string &s)
{
    // remove \r
    s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());

    // cut at first sentence end
    size_t cut = find_first_sentence_end_zh(s);
    if (cut != std::string::npos)
        s.resize(cut);

    // trim leading/trailing spaces/newlines
    auto is_ws = [](unsigned char c)
    { return c == ' ' || c == '\t' || c == '\n'; };
    while (!s.empty() && is_ws((unsigned char)s.front()))
        s.erase(s.begin());
    while (!s.empty() && is_ws((unsigned char)s.back()))
        s.pop_back();

    // collapse internal newlines to space (shouldn't exist after cut, but safe)
    for (char &c : s)
    {
        if (c == '\n' || c == '\t')
            c = ' ';
    }
    // collapse double spaces
    std::string out;
    out.reserve(s.size());
    bool prev_space = false;
    for (char c : s)
    {
        bool sp = (c == ' ');
        if (sp)
        {
            if (!prev_space)
                out.push_back(c);
        }
        else
        {
            out.push_back(c);
        }
        prev_space = sp;
    }
    s.swap(out);
}

int main(int argc, char **argv)
{
    win32_enable_utf8_console();

    std::string model_path;
    std::string user_question = u8"用一句话解释 LR(0) 项目集。";

    int n_predict = 64; // M1.5：默认更短
    int n_ctx = 2048;
    int n_batch = 512;
    float temp = 0.2f; // 短答更稳
    int top_k = 40;
    float top_p = 0.9f;
    int seed = 42;
    bool debug_prompt = false;

    // stop strings: cut off chat leakage / template residue
    std::vector<std::string> stops = {
        "\nHuman:", "\nUser:", "\nassistant:", "\nAssistant:",
        "<|endoftext|>", "</s>", "<|im_end|>", "<|eot_id|>",
        "\n\n"};

    // 加：中文句子终止符（生成到第一句就停）
    // 注意：这里是“末尾出现”判定；真正强制只保留一句话，靠后处理 normalize_one_sentence()
    stops.push_back(u8"。");
    stops.push_back(u8"！");
    stops.push_back(u8"？");
    stops.push_back("\n");

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];

        if (a == "--model" || a == "-m")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --model\n";
                return 2;
            }
            model_path = v;
        }
        else if (a == "--prompt" || a == "-p")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --prompt\n";
                return 2;
            }
            user_question = v;
        }
        else if (a == "--n" || a == "-n")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --n\n";
                return 2;
            }
            n_predict = std::atoi(v);
        }
        else if (a == "--ctx")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --ctx\n";
                return 2;
            }
            n_ctx = std::atoi(v);
        }
        else if (a == "--batch")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --batch\n";
                return 2;
            }
            n_batch = std::atoi(v);
        }
        else if (a == "--temp")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --temp\n";
                return 2;
            }
            temp = (float)std::atof(v);
        }
        else if (a == "--topk")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --topk\n";
                return 2;
            }
            top_k = std::atoi(v);
        }
        else if (a == "--topp")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --topp\n";
                return 2;
            }
            top_p = (float)std::atof(v);
        }
        else if (a == "--seed")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --seed\n";
                return 2;
            }
            seed = std::atoi(v);
        }
        else if (a == "--debug-prompt")
        {
            debug_prompt = true;
        }
        else if (a == "--help" || a == "-h")
        {
            std::cout
                << "Usage:\n"
                << "  llm_cli --model <path.gguf> [--prompt <text>] [--n <tokens>] [--ctx <n>] [--batch <n>]\n"
                << "          [--temp <f>] [--topk <k>] [--topp <p>] [--seed <n>] [--debug-prompt]\n\n"
                << "Example:\n"
                << "  llm_cli --model models\\qwen2.5-3b-instruct-q5_k_m.gguf --prompt \"用一句话解释LR(0)项目集\" --n 48 --temp 0.2\n";
            return 0;
        }
    }

    if (model_path.empty())
    {
        std::cerr << "Error: --model is required\n";
        return 2;
    }

    // 1) init backend
    llama_backend_init();

    // 2) load model
    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model)
    {
        std::cerr << "Failed to load model: " << model_path << "\n";
        llama_backend_free();
        return 3;
    }

    // 3) context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_batch;

    llama_context *ctx = llama_new_context_with_model(model, cparams);
    if (!ctx)
    {
        std::cerr << "Failed to create context\n";
        llama_free_model(model);
        llama_backend_free();
        return 3;
    }

    // 4) 强约束：定义式 + 一句话 + 只输出答案
    // 关键：不要再让它“解释规则/背景”，把输出格式卡死。
    std::string system = u8"你是计算机专业课程助教，只能用中文回答。"
                         u8"输出必须满足："
                         u8"（1）只输出一句话；（2）必须是定义式；（3）不得出现“好的/请/根据/无法/示例”等套话；"
                         u8"（4）不得输出换行；（5）不得输出多余标点。";
    // 把“答案格式”也固定，降低跑偏概率
    std::string user = u8"请按以下格式回答：\n"
                       u8"【定义】LR(0)项目集：<一句话定义>。\n"
                       u8"问题：" +
                       user_question;

    std::vector<llama_chat_message> msgs;
    msgs.push_back({"system", system.c_str()});
    msgs.push_back({"user", user.c_str()});

    std::string prompt;
    prompt.resize(64 * 1024);

    int pn = llama_chat_apply_template(
        nullptr, // use template stored in GGUF metadata
        msgs.data(),
        (int)msgs.size(),
        true, // add assistant prefix
        prompt.data(),
        (int)prompt.size());
    if (pn < 0)
    {
        std::cerr << "llama_chat_apply_template failed\n";
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 6;
    }
    prompt.resize(pn);

    if (debug_prompt)
    {
        std::cerr << "\n[DEBUG PROMPT]\n"
                  << prompt.substr(0, 1200) << "\n[/DEBUG PROMPT]\n";
    }

    // 5) tokenize (parse_special=false 更稳)
    const llama_vocab *vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens;
    tokens.resize(prompt.size() + 64);

    int n_prompt = llama_tokenize(
        vocab,
        prompt.c_str(),
        (int)prompt.size(),
        tokens.data(),
        (int)tokens.size(),
        true, // add_special
        false // parse_special
    );
    if (n_prompt < 0)
    {
        std::cerr << "Tokenize failed\n";
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 4;
    }
    tokens.resize(n_prompt);

    // 6) eval prompt
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    batch.n_tokens = 0;

    for (int i = 0; i < (int)tokens.size(); ++i)
    {
        batch.token[batch.n_tokens] = tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.logits[batch.n_tokens] = false;
        batch.n_tokens++;
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0)
    {
        std::cerr << "llama_decode(prompt) failed\n";
        llama_batch_free(batch);
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 5;
    }

    // 7) sampler chain
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    // dist sampler：seed 用来让输出更可复现
    llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t)seed));

    // 8) generation
    std::cout << "\n--- model output ---\n";

    int n_cur = (int)tokens.size();
    std::string out;
    out.reserve((size_t)n_predict * 6);

    for (int i = 0; i < n_predict; ++i)
    {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, id);

        if (id == llama_token_eos(vocab))
            break;

        char buf[4096];
        int nb = llama_token_to_piece(vocab, id, buf, (int)sizeof(buf), 0, true);
        if (nb <= 0)
            break;

        out.append(buf, buf + nb);

        // stop early
        if (ends_with_any(out, stops))
        {
            trim_at_stop_first_occurrence(out, stops);
            break;
        }

        // feed back
        batch.n_tokens = 0;
        batch.token[batch.n_tokens] = id;
        batch.pos[batch.n_tokens] = n_cur++;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.logits[batch.n_tokens] = true;
        batch.n_tokens++;

        if (llama_decode(ctx, batch) != 0)
            break;
    }

    // final cleanup & enforce one-sentence
    trim_at_stop_first_occurrence(out, stops);
    normalize_one_sentence(out);

    // 如果模型没按格式输出，做一次轻度补救：去掉前导“【定义】”以外的废话
    // （不做复杂规则，避免把正确内容误删）
    // 保留从“LR(0)项目集”开始（如果存在）
    {
        const std::string key = u8"LR(0)";
        size_t p = out.find(key);
        if (p != std::string::npos && p > 0)
        {
            out = out.substr(p);
            normalize_one_sentence(out);
        }
    }

    std::cout << out << "\n--- end ---\n";

    // 9) cleanup
    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
