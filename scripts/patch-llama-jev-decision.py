#!/usr/bin/env python3
"""
patch-llama-jev-decision.py — adds the `jev` decision model type (autotrust/JEV, Atlas3D/JEV-27B-VL-GGUF) to llama.cpp.

llama.cpp b12111 answers `unsupported decision model type: jev` for a GGUF with `<arch>.decision.type = jev`: the
`systemone` server (server-decision) knows openjev, lev, kev, nimble, laya, clef and others, not jev.

The change is the server part of llama.cpp-jev.patch published with Atlas3D/JEV-27B-VL-GGUF (built against upstream
bed0a85), ported by hand to the b12111 tree: the converter, gguf-py and test hunks are not needed to load an existing
GGUF and are left out. It is model independent: everything is read from the GGUF keys `<arch>.decision.*`.

What JEV does at run time:
  * the answer is read from the logits at the last prompt token, like openjev, but per question kind (noul 2,
    score 6, choice 16+) from the verbalizer token ids in `<arch>.decision.verbalizer_ids.<kind>`;
  * `<arch>.decision.bias.<kind>` is added to the logits and `<arch>.decision.temperature.<kind>` divides them
    before the softmax (temperatures are doubles: they are calibrated values);
  * the state is a string, an object or a list of text/image parts; a state object is serialized like Python
    `json.dumps` so that the prompt tokens match the reference implementation.

Every edit is an exact anchor that must occur once; if one is missing the patch fails closed and changes nothing.
"""

import sys
from pathlib import Path

PATCH_GUARD = "COMMON_DECISION_TYPE_JEV"
GUARD_FILE = Path("common") / "common.h"

# (file, anchor, replacement); anchor must occur exactly once in the file
EDITS = []


def edit(path, old, new):
    EDITS.append((Path(path), old, new))


# ---- common: the type and its GGUF name -------------------------------------------------------------------------------

edit("common/common.h",
     "    COMMON_DECISION_TYPE_UNKNOWN, // a decision model of a type that is not supported\n",
     "    COMMON_DECISION_TYPE_JEV,     // calibrated logits of per-kind verbalizer tokens, read at the last prompt token\n"
     "    COMMON_DECISION_TYPE_UNKNOWN, // a decision model of a type that is not supported\n")

edit("common/common.cpp",
     '    { COMMON_DECISION_TYPE_OPENJEV,        "openjev"       },\n',
     '    { COMMON_DECISION_TYPE_OPENJEV,        "openjev"       },\n'
     '    { COMMON_DECISION_TYPE_JEV,            "jev"           },\n')

edit("tools/server/server-common.cpp",
     "        case COMMON_DECISION_TYPE_OPENJEV:\n        case COMMON_DECISION_TYPE_LEV:\n        case COMMON_DECISION_TYPE_KEV:\n"
     "        case COMMON_DECISION_TYPE_NIMBLE:\n        case COMMON_DECISION_TYPE_LAYA:\n",
     "        case COMMON_DECISION_TYPE_OPENJEV:\n        case COMMON_DECISION_TYPE_JEV:\n        case COMMON_DECISION_TYPE_LEV:\n"
     "        case COMMON_DECISION_TYPE_KEV:\n        case COMMON_DECISION_TYPE_NIMBLE:\n        case COMMON_DECISION_TYPE_LAYA:\n")

# ---- server-decision.h ------------------------------------------------------------------------------------------------

edit("tools/server/server-decision.h",
     "            case COMMON_DECISION_TYPE_OPENJEV:\n            case COMMON_DECISION_TYPE_GRANITE_SWITCH:\n",
     "            case COMMON_DECISION_TYPE_OPENJEV:\n            case COMMON_DECISION_TYPE_JEV:\n            case COMMON_DECISION_TYPE_GRANITE_SWITCH:\n")

edit("tools/server/server-decision.h",
     "            case COMMON_DECISION_TYPE_OPENJEV:\n            case COMMON_DECISION_TYPE_CLEF:\n",
     "            case COMMON_DECISION_TYPE_OPENJEV:\n            case COMMON_DECISION_TYPE_JEV:\n            case COMMON_DECISION_TYPE_CLEF:\n")

edit("tools/server/server-decision.h",
     "    std::map<std::string, float> temperatures;",
     "    std::map<std::string, double> temperatures;")

edit("tools/server/server-decision.h",
     "    std::map<size_t, std::vector<float>> score_bias;",
     "    // JEV: the verbalizer token and the bias of each output, by question kind\n"
     "    std::map<server_decision_question_type, std::vector<llama_token>> verbalizer_ids;\n"
     "    std::map<server_decision_question_type, std::vector<double>> biases;\n"
     "    std::map<size_t, std::vector<float>> score_bias;")

edit("tools/server/server-decision.h",
     "    float get_temperature(const server_decision_question & question) const;",
     "    double get_temperature(const server_decision_question & question) const;")

# ---- server-decision.cpp ----------------------------------------------------------------------------------------------

edit("tools/server/server-decision.cpp",
     "#include <cctype>\n",
     "#include <cctype>\n#include <charconv>\n")

edit("tools/server/server-decision.cpp",
     '''    return n < 0 ? "" : std::string(buf);
}
''',
     '''    return n < 0 ? "" : std::string(buf);
}

static std::string decision_jev_meta_str(const llama_model * model, const std::string & key) {
    char buf[2048];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    if (n < 0 || (size_t) n >= sizeof(buf)) {
        throw std::runtime_error("missing or oversized JEV metadata: " + key);
    }
    return std::string(buf, n);
}

static json decision_jev_meta_array(const llama_model * model, const std::string & key, size_t count) {
    const json value = json::parse(decision_jev_meta_str(model, key));
    if (!value.is_array() || value.size() != count) {
        throw std::runtime_error("invalid JEV metadata array: " + key);
    }
    return value;
}
''')

# JEV has its own strict temperatures (doubles, one per kind): skip the generic float loop
edit("tools/server/server-decision.cpp",
     '''    for (int32_t i = 0; i < llama_model_meta_count(model); i++) {
        char key[256];
        char val[64];
        if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix_temp)) {''',
     '''    for (int32_t i = 0; i < llama_model_meta_count(model); i++) {
        if (model_type == COMMON_DECISION_TYPE_JEV) {
            break;
        }
        char key[256];
        char val[64];
        if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix_temp)) {''')

edit("tools/server/server-decision.cpp",
     '''    if (model_type == COMMON_DECISION_TYPE_OPENJEV) {
        // one letter per option, each must be a single token
''',
     '''    if (model_type == COMMON_DECISION_TYPE_JEV) {
        for (const auto kind : {SERVER_DECISION_QUESTION_NOUL, SERVER_DECISION_QUESTION_SCORE, SERVER_DECISION_QUESTION_CHOICE}) {
            const std::string name = decision_question_type_name(kind);
            const size_t count = kind == SERVER_DECISION_QUESTION_NOUL ? 2 : kind == SERVER_DECISION_QUESTION_SCORE ? 6 : 16;
            const auto ids = decision_jev_meta_array(model, prefix + "verbalizer_ids." + name, count);
            const auto bias = decision_jev_meta_array(model, prefix + "bias." + name, count);
            for (size_t i = 0; i < count; i++) {
                if (!ids.at(i).is_number_integer()) {
                    throw std::runtime_error("invalid JEV verbalizer token: " + name);
                }
                const int64_t id = ids.at(i).get<int64_t>();
                if (id < 0 || id >= llama_vocab_n_tokens(vocab) || !bias.at(i).is_number()) {
                    throw std::runtime_error("invalid JEV verbalizer or bias: " + name);
                }
                const double b = bias.at(i).get<double>();
                if (!std::isfinite(b)) {
                    throw std::runtime_error("invalid JEV bias: " + name);
                }
                verbalizer_ids[kind].push_back((llama_token) id);
                biases[kind].push_back(b);
            }
            const std::string val = decision_jev_meta_str(model, prefix_temp + name);
            char * end = nullptr;
            const double temp = std::strtod(val.c_str(), &end);
            if (end == val.c_str() || *end != '\\0' || !std::isfinite(temp) || temp <= 0.0) {
                throw std::runtime_error("invalid JEV temperature: " + name);
            }
            temperatures[name] = temp;
        }
        // choice labels: A..Z, then AA..ZZ, only the codes that are a single token, also inside a text
        std::vector<std::string> codes;
        for (char a = 'A'; a <= 'Z'; a++) {
            codes.push_back(std::string(1, a));
        }
        for (char a = 'A'; a <= 'Z'; a++) {
            for (char b = 'A'; b <= 'Z'; b++) {
                codes.push_back(std::string{a, b});
            }
        }
        for (const auto & code : codes) {
            const auto toks = common_tokenize(vocab, code, false, false);
            const auto contextual = common_tokenize(vocab, "x\\n" + code + ") y", false, false);
            if (toks.size() == 1 && std::find(contextual.begin(), contextual.end(), toks[0]) != contextual.end()) {
                labels.push_back(toks[0]);
                label_texts.push_back(code);
            }
            if (labels.size() == 256) {
                break;
            }
        }
        const auto & trained = verbalizer_ids.at(SERVER_DECISION_QUESTION_CHOICE);
        if (labels.size() < trained.size() || !std::equal(trained.begin(), trained.end(), labels.begin())) {
            throw std::runtime_error("JEV choice labels do not match the trained verbalizers");
        }
        n_options_max = labels.size();
    } else if (model_type == COMMON_DECISION_TYPE_OPENJEV) {
        // one letter per option, each must be a single token
''')

edit("tools/server/server-decision.cpp",
     "        question.instructions = instructions;\n",
     "        question.instructions = instructions;\n"
     "        if (type == COMMON_DECISION_TYPE_JEV && !question.instructions.is_string()) {\n"
     '            throw err("\\"instructions\\" must be a string for JEV");\n'
     "        }\n")

edit("tools/server/server-decision.cpp",
     "                question.options.push_back({key, description});\n            }\n            if (strict_payload && question.options.size() < 2) {\n",
     "                if (type == COMMON_DECISION_TYPE_JEV && !description.is_null() && !description.is_string()) {\n"
     '                    throw err("choice descriptions must be strings or null for JEV");\n'
     "                }\n"
     "                question.options.push_back({key, description});\n            }\n"
     "            if (type == COMMON_DECISION_TYPE_JEV && question.options.size() < 2) {\n"
     '                throw err("JEV choice needs at least two options");\n'
     "            }\n"
     "            if (strict_payload && question.options.size() < 2) {\n")

edit("tools/server/server-decision.cpp",
     '''                throw err("\\"criteria\\" must be an array of 2 to 10 levels");
            }
''',
     '''                throw err("\\"criteria\\" must be an array of 2 to 10 levels");
            }
            if (type == COMMON_DECISION_TYPE_JEV && criteria.size() != 6) {
                throw err("JEV score requires exactly six levels");
            }
''')

JEV_STATE = '''// Python json.dumps notation for the state object of a JEV prompt: ", " and ": " separators, floats like repr()
static std::string decision_jev_json_text(const json & value) {
    if (value.is_object()) {
        std::string text = "{";
        for (const auto & [key, item] : value.items()) {
            text += (text.size() == 1 ? "" : ", ") + json(key).dump() + ": " + decision_jev_json_text(item);
        }
        return text + "}";
    }
    if (value.is_array()) {
        std::string text = "[";
        for (const auto & item : value) {
            text += (text.size() == 1 ? "" : ", ") + decision_jev_json_text(item);
        }
        return text + "]";
    }
    if (!value.is_number_float()) {
        return value.dump();
    }
    char buf[64];
    const auto result = std::to_chars(buf, buf + sizeof(buf), value.get<double>(), std::chars_format::scientific);
    if (result.ec != std::errc()) {
        throw std::invalid_argument("cannot serialize JEV state number");
    }
    const std::string scientific(buf, result.ptr);
    const size_t e = scientific.find('e');
    const int exponent = std::stoi(scientific.substr(e + 1));
    if (exponent < -4 || exponent >= 16) {
        return scientific;
    }
    const bool negative = scientific[0] == '-';
    std::string digits = scientific.substr(negative ? 1 : 0, e - (negative ? 1 : 0));
    string_replace_all(digits, ".", "");
    std::string text = negative ? "-" : "";
    const int decimal = exponent + 1;
    if (decimal <= 0) {
        return text + "0." + std::string(-decimal, '0') + digits;
    }
    if ((size_t) decimal >= digits.size()) {
        return text + digits + std::string(decimal - digits.size(), '0') + ".0";
    }
    return text + digits.substr(0, decimal) + "." + digits.substr(decimal);
}

json server_decision_context::parse_state(const json & body, std::vector<raw_buffer> & files) const {
    if (type == COMMON_DECISION_TYPE_JEV) {
        // the state is a string, an object or a list of parts; the template gets a list of text and image parts
        const json & state = body.at("state");
        if (!state.is_string() && !state.is_object() && !state.is_array()) {
            throw std::invalid_argument("JEV state must be a string, object or array of parts");
        }
        json parts = json::array();
        auto add_image = [&](const json & url) {
            const size_t index = files.size();
            decision_load_image(url, files);
            parts.push_back({{"type", "image_url"}, {"image_index", index}});
        };
        const json input_parts = state.is_array() ? state : json::array({state});
        for (const auto & part : input_parts) {
            if (state.is_array() && part.is_object() && part.contains("image")) {
                add_image(part.at("image"));
            } else if (state.is_array() && part.is_object() && json_value(part, "type", std::string()) == "image_url") {
                if (!part.contains("image_url")) {
                    throw std::invalid_argument("image_url part must contain image_url");
                }
                const json & url = part.at("image_url");
                add_image(url.is_object() && url.contains("url") ? url.at("url") : url);
            } else if (state.is_array() && part.is_object() && json_value(part, "type", std::string()) == "text") {
                if (!part.contains("text") || !part.at("text").is_string()) {
                    throw std::invalid_argument("text part must contain a string text");
                }
                parts.push_back({{"type", "text"}, {"text", part.at("text")}});
            } else {
                parts.push_back({{"type", "text"}, {"text", part.is_string() ? part.get<std::string>() : decision_jev_json_text(part)}});
            }
        }
        // "images" is an alias of "files"
        for (const char * key : {"files", "images"}) {
            if (!body.contains(key) || body.at(key).is_null()) {
                continue;
            }
            if (!body.at(key).is_array()) {
                throw std::invalid_argument(std::string("\\"") + key + "\\" must be an array");
            }
            for (const auto & url : body.at(key)) {
                add_image(url);
            }
        }
        return parts;
    }
'''
edit("tools/server/server-decision.cpp",
     "json server_decision_context::parse_state(const json & body, std::vector<raw_buffer> & files) const {\n",
     JEV_STATE)

edit("tools/server/server-decision.cpp",
     "    if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_GRANITE_SWITCH || type == COMMON_DECISION_TYPE_LEV",
     '''    if (type == COMMON_DECISION_TYPE_JEV) {
        // the verbalizers depend on the kind of the question; choice uses the label codes
        const auto & ids = question.type == SERVER_DECISION_QUESTION_CHOICE ? labels : verbalizer_ids.at(question.type);
        task.decision.labels.assign(ids.begin(), ids.begin() + n_outputs(question));
        if (!media.empty()) {
            task.tokens = process_mtmd_prompt(mctx, prompt, media, init_opt);
            return;
        }
    }
    if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_GRANITE_SWITCH || type == COMMON_DECISION_TYPE_LEV''')

edit("tools/server/server-decision.cpp",
     "float server_decision_context::get_temperature(",
     "double server_decision_context::get_temperature(")

edit("tools/server/server-decision.cpp",
     "    float temperature = has_media && type == COMMON_DECISION_TYPE_LFM2_D1_OMNI ? 1.0f : get_temperature(question);",
     "    double temperature = has_media && type == COMMON_DECISION_TYPE_LFM2_D1_OMNI ? 1.0 : get_temperature(question);")

edit("tools/server/server-decision.cpp",
     "        const double logit_max = *std::max_element(logits.begin(), logits.end());\n",
     "        if (type == COMMON_DECISION_TYPE_JEV) {\n"
     "            // calibration offsets of the trained verbalizers\n"
     "            const auto & bias = biases.at(question.type);\n"
     "            for (size_t i = 0; i < std::min(n, bias.size()); i++) {\n"
     "                logits[i] += bias[i];\n"
     "            }\n"
     "        }\n"
     "        const double logit_max = *std::max_element(logits.begin(), logits.end());\n")


def find_sources(root: Path):
    out = []
    for variant in ("cuda_v12", "cuda_v13", "cuda_v11"):
        base = root / "build" / f"llama-server-{variant}" / "_deps" / "llama_cpp-src"
        if (base / GUARD_FILE).is_file():
            out.append(base)
    return out


def patch(base: Path) -> bool:
    if PATCH_GUARD in (base / GUARD_FILE).read_text():
        print(f"  Already patched: {base}")
        return True
    contents = {}
    for rel, old, new in EDITS:
        text = contents.get(rel)
        if text is None:
            text = (base / rel).read_text()
        if text.count(old) != 1:
            print(f"  ERROR: anchor not found exactly once in {base / rel}:\n{old[:200]}", file=sys.stderr)
            return False
        contents[rel] = text.replace(old, new, 1)
    for rel, text in contents.items():  # all anchors matched: write
        (base / rel).write_text(text)
    print(f"  jev decision type applied to {base}")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        return 2
    bases = find_sources(Path(sys.argv[1]))
    if not bases:
        print("ERROR: common/common.h not found in any build/llama-server-cuda_v* tree", file=sys.stderr)
        return 1
    return 0 if all(patch(b) for b in bases) else 1


if __name__ == "__main__":
    sys.exit(main())
