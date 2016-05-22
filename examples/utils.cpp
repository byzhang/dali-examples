#include "utils.h"

#include <dali/array/memory/device.h>
#include <dali/runtime_config.h>
#include <dali/utils/assert2.h>
#include <dali/utils/print_utils.h>

DEFINE_string(visualizer_hostname, "127.0.0.1", "Default hostname to be used by visualizer.");
DEFINE_int32(visualizer_port,      6379,        "Default port to be used by visualizer.");
DEFINE_string(visualizer,          "",          "What to name the visualization job.");
DEFINE_int32(device,              -1,           "What device to run the computation on (-1 for cpu, or number of gpu: 0, 1, ..).");

using std::string;
using std::stringstream;
using std::vector;

namespace utils {

    template<typename T>
    void load_corpus_from_stream(Corpus& corpus, T& stream) {
        corpus.ParseFromIstream(&stream);
    }

    template void load_corpus_from_stream(Corpus&, igzstream&);
    template void load_corpus_from_stream(Corpus&, std::fstream&);
    template void load_corpus_from_stream(Corpus&, std::stringstream&);
    template void load_corpus_from_stream(Corpus&, std::istream&);

    Corpus load_corpus_protobuff(const std::string& path) {
        Corpus corpus;
        if (is_gzip(path)) {
            igzstream fpgz(path.c_str(), std::ios::in | std::ios::binary);
            load_corpus_from_stream(corpus, fpgz);
        } else {
            std::fstream fp(path, std::ios::in | std::ios::binary);
            load_corpus_from_stream(corpus, fp);
        }
        return corpus;
    }


    tokenized_labeled_dataset load_protobuff_dataset(string directory, const vector<string>& index2label) {
        ensure_directory(directory);
        auto files = listdir(directory);
        tokenized_labeled_dataset dataset;
        for (auto& file : files) {
            auto corpus = load_corpus_protobuff(directory + file);
            for (auto& example : corpus.example()) {
                dataset.emplace_back(std::initializer_list<vector<string>>({
                    vector<string>(example.words().begin(), example.words().end()),
                    triggers_to_strings(example.trigger(), index2label)
                }));
            }
        }
        return dataset;
    }


    tokenized_labeled_dataset load_protobuff_dataset(
        SQLite::Statement& query,
        const vector<string>& index2label,
        int num_elements,
        int column) {
        int els_seen = 0;
        tokenized_labeled_dataset dataset;
        while (query.executeStep()) {
            const char* protobuff_serialized = query.getColumn(column);
            stringstream ss(protobuff_serialized);
            Corpus corpus;
            load_corpus_from_stream(corpus, ss);
            for (auto& example : corpus.example()) {
                dataset.emplace_back(std::initializer_list<vector<string>>{
                    vector<string>(example.words().begin(), example.words().end()),
                    triggers_to_strings(example.trigger(), index2label)
                });
                ++els_seen;
            }
            if (els_seen >= num_elements) {
                break;
            }
        }
        return dataset;
    }

    str_sequence triggers_to_strings(const google::protobuf::RepeatedPtrField<Example::Trigger>& triggers, const str_sequence& index2target) {
        str_sequence data;
        data.reserve(triggers.size());
        for (auto& trig : triggers)
            if (trig.id() < index2target.size())
                data.emplace_back(index2target[trig.id()]);
        return data;
    }

    void update_device(int device_name) {
        if (device_name == -1) {
            memory::default_preferred_device = memory::Device::cpu();
        } else if (device_name >= 0) {
            #ifdef DALI_USE_CUDA
                memory::default_preferred_device = memory::Device::gpu(device_name);
            #else
                utils::assert2(
                    device_name >= -1,
                    utils::MS() << "Dali compiled without GPU support: "
                                << "update_device's device_name argument must be -1 for cpu"
                );

            #endif
        } else {
            utils::assert2(
                device_name >= -1,
                utils::MS() << "update_device's device_name argument must be >= -1"
            );
        }
    }
}
