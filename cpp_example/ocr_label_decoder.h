#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vision::ocr {

struct LabelDecodeResult {
    std::string text;
    bool had_invalid_label = false;
    std::int64_t first_invalid_label = 0;
};

inline LabelDecodeResult DecodeLabelSequence(const std::vector<std::int64_t>& labels)
{
    // Index 0 = CTC blank; indices 1.. map to model_meta.json "charset" in order.
    // data_1 5MB charset is "dlru", so 1=d, 2=l, 3=r, 4=u.
    static constexpr const char* LabelMap[] = { "", "d", "l", "r", "u" };
    static constexpr std::int64_t LabelCount = static_cast<std::int64_t>(sizeof(LabelMap) / sizeof(LabelMap[0]));

    LabelDecodeResult result;
    std::int64_t last_label = 0;
    for (const std::int64_t label : labels) {
        if (label == last_label) {
            continue;
        }
        last_label = label;
        if (label == 0) {
            continue;
        }
        if (label < 0 || label >= LabelCount) {
            if (!result.had_invalid_label) {
                result.first_invalid_label = label;
            }
            result.had_invalid_label = true;
            continue;
        }
        result.text += LabelMap[label];
    }
    return result;
}

} // namespace vision::ocr
