#pragma once

enum class FilterSize {
    K3x3 = 3,
    K5x5 = 5,
    K7x7 = 7
};

enum class BorderPolicy {
    NO_BORDER,
    REPLICATE,
    CONSTANT,
    MIRROR
};

struct BorderMode {
    BorderPolicy policy;
    int value = 0;
};
