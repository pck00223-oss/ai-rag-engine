// apps/llm_cli.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>

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

// 只有当你要用 --db/--ids 从 SQLite 取证据时才需要
#include <sqlite3.h>

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
    size_t nl = s.find('\n');
    if (nl != std::string::npos)
    {
        best = (best == std::string::npos) ? nl : std::min(best, nl);
    }
    return best;
}

static void normalize_one_sentence(std::string &s)
{
    s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());

    size_t cut = find_first_sentence_end_zh(s);
    if (cut != std::string::npos)
        s.resize(cut);

    auto is_ws = [](unsigned char c)
    { return c == ' ' || c == '\t' || c == '\n'; };
    while (!s.empty() && is_ws((unsigned char)s.front()))
        s.erase(s.begin());
    while (!s.empty() && is_ws((unsigned char)s.back()))
        s.pop_back();

    for (char &c : s)
    {
        if (c == '\n' || c == '\t')
            c = ' ';
    }

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

// ---------- file utils ----------
static bool read_all_text(const std::string &path, std::string &out)
{
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs)
        return false;
    std::ostringstream ss;
    ss << ifs.rdbuf();
    out = ss.str();
    return true;
}

static std::vector<int64_t> parse_ids_csv(const std::string &s)
{
    std::vector<int64_t> ids;
    std::string cur;
    for (char ch : s)
    {
        if (ch == ',' || ch == ';' || ch == ' ')
        {
            if (!cur.empty())
            {
                ids.push_back(std::stoll(cur));
                cur.clear();
            }
        }
        else
        {
            cur.push_back(ch);
        }
    }
    if (!cur.empty())
        ids.push_back(std::stoll(cur));
    return ids;
}

// ---------- SQLite evidence loader (by ids) ----------
// 假设你的表结构是：documents(id INTEGER PRIMARY KEY, filename TEXT, content TEXT)
// 或者 chunks(id INTEGER PRIMARY KEY, doc TEXT, chunk_idx INT, content TEXT)
// 你可以通过 --table 指定表名（默认 documents），--col 指定内容列（默认 content）。
static std::string load_context_from_sqlite_by_ids(
    const std::string &db_path,
    const std::string &table,
    const std::string &content_col,
    const std::vector<int64_t> &ids)
{
    if (ids.empty())
        return "";

    sqlite3 *db = nullptr;
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK)
    {
        if (db)
            sqlite3_close(db);
        return "";
    }

    // 构造 IN (?, ?, ?) 占位符
    std::ostringstream ph;
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (i)
            ph << ",";
        ph << "?";
    }

    std::string sql =
        "SELECT id, " + content_col +
        " FROM " + table +
        " WHERE id IN (" + ph.str() + ")";

    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK)
    {
        sqlite3_close(db);
        return "";
    }

    for (size_t i = 0; i < ids.size(); ++i)
    {
        sqlite3_bind_int64(stmt, (int)i + 1, ids[i]);
    }

    std::ostringstream ctx;
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        int64_t id = sqlite3_column_int64(stmt, 0);
        const unsigned char *content = sqlite3_column_text(stmt, 1);
        if (content)
        {
            ctx << "[证据#" << id << "] " << (const char *)content << "\n";
        }
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return ctx.str();
}

int main(int argc, char **argv)
{
    win32_enable_utf8_console();

    std::string model_path;
    std::string user_question = u8"用一句话解释 LR(0) 项目集。";

    // M1：把证据上下文“注入”prompt（来自文件或SQLite）
    std::string context_file;               // --context-file
    std::string sqlite_db;                  // --db
    std::string sqlite_table = "documents"; // --table
    std::string sqlite_col = "content";     // --col
    std::string ids_csv;                    // --ids

    int n_predict = 64;
    int n_ctx = 2048;
    int n_batch = 512;
    float temp = 0.2f;
    int top_k = 40;
    float top_p = 0.9f;
    int seed = 42;
    bool debug_prompt = false;

    std::vector<std::string> stops = {
        "\nHuman:", "\nUser:", "\nassistant:", "\nAssistant:",
        "<|endoftext|>", "</s>", "<|im_end|>", "<|eot_id|>",
        "\n\n"};
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
        else if (a == "--context-file")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --context-file\n";
                return 2;
            }
            context_file = v;
        }
        else if (a == "--db")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --db\n";
                return 2;
            }
            sqlite_db = v;
        }
        else if (a == "--table")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --table\n";
                return 2;
            }
            sqlite_table = v;
        }
        else if (a == "--col")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --col\n";
                return 2;
            }
            sqlite_col = v;
        }
        else if (a == "--ids")
        {
            const char *v = get_arg(i, argc, argv);
            if (!v)
            {
                std::cerr << "Missing value for --ids\n";
                return 2;
            }
            ids_csv = v;
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
                << "  llm_cli --model <path.gguf> [--prompt <text>]\n"
                << "          [--context-file <context.txt>]\n"
                << "          [--db <documents.db> --table <table> --col <content_col> --ids 1,2,3]\n"
                << "          [--n <tokens>] [--ctx <n>] [--batch <n>]\n"
                << "          [--temp <f>] [--topk <k>] [--topp <p>] [--seed <n>] [--debug-prompt]\n\n"
                << "Examples:\n"
                << "  llm_cli --model models\\qwen2.5-3b-instruct-q5_k_m.gguf --prompt \"解释LR(0)项目集\" --context-file context.txt\n"
                << "  llm_cli --model models\\qwen2.5-3b-instruct-q5_k_m.gguf --prompt \"...\" --db documents.db --table documents --col content --ids 1,2,3\n";
            return 0;
        }
    }

    if (model_path.empty())
    {
        std::cerr << "Error: --model is required\n";
        return 2;
    }

    // ------- 0) load evidence context -------
    std::string evidence;
    if (!context_file.empty())
    {
        if (!read_all_text(context_file, evidence))
        {
            std::cerr << "Warning: failed to read context-file: " << context_file << "\n";
        }
    }
    else if (!sqlite_db.empty() && !ids_csv.empty())
    {
        auto ids = parse_ids_csv(ids_csv);
        evidence = load_context_from_sqlite_by_ids(sqlite_db, sqlite_table, sqlite_col, ids);
        if (evidence.empty())
        {
            std::cerr << "Warning: no evidence loaded from sqlite (check db/table/col/ids).\n";
        }
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

    // 4) system + user prompt (注入证据)
    std::string system = u8"你是计算机专业课程助教，只能用中文回答。"
                         u8"输出必须满足："
                         u8"（1）只输出一句话；（2）必须是定义式；（3）不得出现“好的/请/根据/无法/示例”等套话；"
                         u8"（4）不得输出换行；（5）不得输出多余标点。";

    std::string user;
    if (!evidence.empty())
    {
        user = u8"以下是检索到的资料证据（回答必须基于这些证据，且不得编造）：\n";
        user += evidence;
        user += u8"\n请按以下格式回答：\n【定义】LR(0)项目集：<一句话定义>。\n问题：";
        user += user_question;
    }
    else
    {
        user = u8"请按以下格式回答：\n【定义】LR(0)项目集：<一句话定义>。\n问题：";
        user += user_question;
    }

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

    // 5) tokenize
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

        if (ends_with_any(out, stops))
        {
            trim_at_stop_first_occurrence(out, stops);
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

    trim_at_stop_first_occurrence(out, stops);
    normalize_one_sentence(out);

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
