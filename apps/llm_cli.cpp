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

static void trim_at_stop(std::string &s, const std::vector<std::string> &stops)
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

static void win32_enable_utf8_console()
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

#ifdef _WIN32
// UTF-16 -> UTF-8
static std::string wide_to_utf8(const wchar_t *w)
{
    if (!w)
        return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 1)
        return {};
    std::string s((size_t)len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), len, nullptr, nullptr);
    return s;
}
#endif

// 把原 main 的内容放到这里
static int real_main(int argc, char **argv)
{
    win32_enable_utf8_console();

    std::string model_path;
    std::string user_prompt = u8"用一句话解释 LR(0) 项目集。";

    int n_predict = 128;
    int n_ctx = 2048;
    int n_batch = 512;
    float temp = 0.2f;
    int top_k = 40;
    float top_p = 0.9f;
    int seed = 42;
    bool debug_prompt = false;

    std::vector<std::string> stops = {
        "\nHuman:", "\nUser:", "\nassistant:", "\nAssistant:",
        "<|endoftext|>", "</s>", "<|im_end|>", "<|eot_id|>"};

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
            user_prompt = v; // 现在这里一定是 UTF-8 了（wmain 转的）
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
                << "  llm_cli --model models\\qwen2.5-3b-instruct-q5_k_m.gguf --prompt \"用一句话解释LR(0)项目集\" --n 64 --temp 0.2\n";
            return 0;
        }
    }

    if (model_path.empty())
    {
        std::cerr << "Error: --model is required\n";
        return 2;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model)
    {
        std::cerr << "Failed to load model: " << model_path << "\n";
        llama_backend_free();
        return 3;
    }

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

    // 用 GGUF 自带 chat template
    std::string system = u8"你是计算机专业课程助教，只能用中文回答。要求：只用一句话，不解释背景，不举例。";
    std::string user = user_prompt;

    std::vector<llama_chat_message> msgs;
    msgs.push_back({"system", system.c_str()});
    msgs.push_back({"user", user.c_str()});

    std::string prompt;
    prompt.resize(32 * 1024);

    int n = llama_chat_apply_template(
        nullptr,
        msgs.data(),
        (int)msgs.size(),
        true,
        prompt.data(),
        (int)prompt.size());
    if (n < 0)
    {
        std::cerr << "llama_chat_apply_template failed\n";
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 6;
    }
    prompt.resize(n);

    if (debug_prompt)
    {
        std::cerr << "\n[DEBUG PROMPT]\n"
                  << prompt << "\n[/DEBUG PROMPT]\n";
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens;
    tokens.resize(prompt.size() + 32);

    int n_prompt = llama_tokenize(
        vocab,
        prompt.c_str(),
        (int)prompt.size(),
        tokens.data(),
        (int)tokens.size(),
        true,
        false);
    if (n_prompt < 0)
    {
        std::cerr << "Tokenize failed\n";
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 4;
    }
    tokens.resize(n_prompt);

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

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(-1));

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

        if (ends_with_any(out, stops))
        {
            trim_at_stop(out, stops);
            break;
        }

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

    trim_at_stop(out, stops);
    std::cout << out << "\n--- end ---\n";

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}

#ifdef _WIN32
// Windows：用 wmain 接收 UTF-16 参数，转成 UTF-8 再跑 real_main
int wmain(int argc, wchar_t **wargv)
{
    std::vector<std::string> argv_utf8;
    argv_utf8.reserve(argc);
    for (int i = 0; i < argc; ++i)
        argv_utf8.push_back(wide_to_utf8(wargv[i]));

    std::vector<char *> argv;
    argv.reserve(argc);
    for (auto &s : argv_utf8)
        argv.push_back(s.data());

    return real_main(argc, argv.data());
}
#else
int main(int argc, char **argv)
{
    return real_main(argc, argv);
}
#endif
