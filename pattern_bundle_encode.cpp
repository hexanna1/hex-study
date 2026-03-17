#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

using Bytes = std::vector<uint8_t>;
using Pair = std::pair<int, int>;

constexpr int kFractionBits = 10;
constexpr int kFractionLimit = 1 << kFractionBits;
constexpr int kRowPredictorNumerator = 3;
constexpr int kRowPredictorDenominator = 4;
constexpr int kRowPredictorIntercept = 84;
constexpr int kSingletonDistance = 16;
constexpr int kNearDistance = 7;
constexpr int kWideDistance = 4;
constexpr std::array<int, 6> kDistanceCutoffs{1, 3, 4, 7, 9, 12};
constexpr int kDistanceClasses = kDistanceCutoffs.size() + 1;
constexpr int kDistanceContexts = kDistanceClasses * kDistanceClasses;
constexpr int kSecondDistanceContexts = kDistanceContexts * kDistanceContexts;
constexpr int kBiasScale = 6;
constexpr std::array<std::array<int, 4>, 12> kTransforms{{
    {1, 0, 0, 1}, {0, -1, 1, 1}, {-1, -1, 1, 0}, {-1, 0, 0, -1},
    {0, 1, -1, -1}, {1, 1, -1, 0}, {1, 0, -1, -1}, {0, -1, -1, 0},
    {-1, -1, 0, 1}, {-1, 0, 1, 1}, {0, 1, 1, 0}, {1, 1, 0, -1},
}};

static void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

class Reader {
public:
  explicit Reader(Bytes bytes) : bytes(std::move(bytes)) {}

  uint8_t u8() {
    require(offset < bytes.size(), "truncated encoder input");
    return bytes[offset++];
  }

  int8_t i8() { return (int8_t)u8(); }

  uint16_t u16() {
    uint16_t value = u8();
    value |= (uint16_t)u8() << 8;
    return value;
  }

  uint32_t u32() {
    uint32_t value = 0;
    for (int i = 0; i < 4; i++)
      value |= (uint32_t)u8() << (8 * i);
    return value;
  }

  Bytes take(size_t count) {
    require(count <= bytes.size() - offset, "truncated encoder input");
    Bytes out(bytes.begin() + (ptrdiff_t)offset, bytes.begin() + (ptrdiff_t)(offset + count));
    offset += count;
    return out;
  }

  bool done() const { return offset == bytes.size(); }

private:
  Bytes bytes;
  size_t offset = 0;
};

static void append_u16(Bytes &out, uint16_t value) {
  out.push_back((uint8_t)value);
  out.push_back((uint8_t)(value >> 8));
}

static void append_u32(Bytes &out, size_t value) {
  require(value <= std::numeric_limits<uint32_t>::max(), "bundle field exceeds u32 range");
  for (int i = 0; i < 4; i++)
    out.push_back((uint8_t)(value >> (8 * i)));
}

static void append(Bytes &out, const Bytes &tail) { out.insert(out.end(), tail.begin(), tail.end()); }

static void append_uvarint(Bytes &out, size_t value) {
  while (value >= 0x80) {
    out.push_back((uint8_t)((value & 0x7f) | 0x80));
    value >>= 7;
  }
  out.push_back((uint8_t)value);
}

static int bit_length(int value) {
  int bits = 0;
  while (value > 0) {
    bits++;
    value >>= 1;
  }
  return bits;
}

static Bytes pack_bitplanes(const std::vector<int> &values, int bits) {
  Bytes out((values.size() * (size_t)bits + 7) / 8, 0);
  size_t offset = 0;
  for (int bit = bits - 1; bit >= 0; bit--)
    for (int value : values) {
      out[offset / 8] |= (uint8_t)(((value >> bit) & 1) << (offset % 8));
      offset++;
    }
  return out;
}

static int round_ratio_half_even(int64_t numerator, int64_t denominator) {
  require(denominator > 0, "nonpositive predictor denominator");
  require(numerator >= 0, "negative predictor numerator");
  int64_t quotient = numerator / denominator;
  int64_t remainder = numerator % denominator;
  if (remainder * 2 > denominator || (remainder * 2 == denominator && (quotient & 1)))
    quotient++;
  return (int)quotient;
}

struct Entry {
  Bytes key;
  int tenuki;
  std::vector<std::pair<Pair, int>> cells;
};

static Pair key_point(uint8_t code) { return {code & 7, (code >> 3) - 8}; }

static uint16_t stabilizer_mask(const Bytes &key) {
  size_t plus_count = key[0] & 15;
  std::vector<Pair> plus, minus;
  for (size_t i = 1; i < key.size(); i++) {
    Pair point = key_point(key[i]);
    (i <= plus_count ? plus : minus).push_back(point);
  }
  uint16_t mask = 0;
  for (size_t index = 1; index < kTransforms.size(); index++) {
    const auto &t = kTransforms[index];
    std::vector<Pair> transformed_plus, transformed_minus;
    Pair anchor{100, 100};
    for (size_t i = 1; i < key.size(); i++) {
      const auto [q, r] = key_point(key[i]);
      Pair point{t[0] * q + t[1] * r, t[2] * q + t[3] * r};
      anchor = std::min(anchor, point);
      (i <= plus_count ? transformed_plus : transformed_minus).push_back(point);
    }
    for (auto *points : {&transformed_plus, &transformed_minus}) {
      for (Pair &point : *points) {
        point.first -= anchor.first;
        point.second -= anchor.second;
      }
      std::sort(points->begin(), points->end());
    }
    if (transformed_plus == plus && transformed_minus == minus)
      mask |= (uint16_t)(1U << index);
  }
  return mask;
}

static int distance_context(const Bytes &key, Pair pair) {
  const size_t plus_count = key[0] & 15;
  const int far = kDistanceCutoffs.back() + 1;
  std::array<int, 2> first{far, far}, second{far, far};
  for (size_t i = 1; i < key.size(); i++) {
    const auto [q, r] = key_point(key[i]);
    const int dq = pair.first - q, dr = pair.second - r;
    const int color = i <= plus_count ? 0 : 1;
    const int distance = dq * dq + dq * dr + dr * dr;
    second[color] = std::min(second[color], std::max(first[color], distance));
    first[color] = std::min(first[color], distance);
  }
  const auto class_for = [](int dist) {
    return (int)(std::lower_bound(kDistanceCutoffs.begin(), kDistanceCutoffs.end(), dist) -
                 kDistanceCutoffs.begin());
  };
  const int nearest = class_for(first[0]) * kDistanceClasses + class_for(first[1]);
  const int next = class_for(second[0]) * kDistanceClasses + class_for(second[1]);
  return nearest * kDistanceContexts + next;
}

static int quantized_median(std::vector<int> &sample) {
  if (sample.empty())
    return 0;
  auto median = sample.begin() + (sample.size() - 1) / 2;
  std::nth_element(sample.begin(), median, sample.end());
  const int magnitude = round_ratio_half_even(std::abs(*median), kBiasScale);
  return std::clamp(*median < 0 ? -magnitude : magnitude, -128, 127);
}

static int stone_fraction_code(int value, int prediction) {
  if (value >= 1001)
    return value - 1001;  // Dead and captured cells.
  if (value >= 999)
    return 2 + 1000 - value;
  prediction = std::clamp(prediction, 0, 998);
  const int delta = value - prediction;
  const int radius = std::min(prediction, 998 - prediction);
  // Alternate signs while both sides have codes, then assign the remaining tail.
  return 4 + (std::abs(delta) <= radius
                  ? 2 * std::abs(delta) - (delta < 0)
                  : std::abs(delta) + radius);
}

static std::vector<Entry> read_entries(uint32_t &corpus_count) {
  Bytes input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  Reader reader(std::move(input));
  require(reader.take(3) == Bytes({'H', 'P', 'I'}), "bad encoder input magic");
  uint32_t count = reader.u32();
  corpus_count = reader.u32();
  std::vector<Entry> entries;
  entries.reserve(count);
  for (uint32_t i = 0; i < count; i++) {
    Entry entry;
    entry.key = reader.take(reader.u8());
    require(!entry.key.empty(), "empty pattern key");
    entry.tenuki = reader.u16();
    require(entry.tenuki < kFractionLimit, "tenuki fraction outside u10 range");
    uint16_t cell_count = reader.u16();
    entry.cells.reserve(cell_count);
    for (uint16_t j = 0; j < cell_count; j++) {
      Pair pair{reader.i8(), reader.i8()};
      int value = reader.u16();
      require(value < kFractionLimit, "stone fraction outside u10 range");
      entry.cells.emplace_back(pair, value);
    }
    std::sort(entry.cells.begin(), entry.cells.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    for (size_t j = 1; j < entry.cells.size(); j++)
      require(entry.cells[j - 1].first != entry.cells[j].first,
              "duplicate pattern coordinate");
    entries.push_back(std::move(entry));
  }
  require(reader.done(), "trailing encoder input");
  std::sort(entries.begin(), entries.end(),
            [](const Entry &a, const Entry &b) { return a.key < b.key; });
  return entries;
}

static Bytes encode(const std::vector<Entry> &entries, uint32_t corpus_count) {
  Bytes key_stream;
  Bytes previous_key;
  std::vector<int> tenuki;
  std::map<Pair, std::vector<std::pair<size_t, int>>> pair_entries;
  size_t cell_count = 0;
  for (size_t row = 0; row < entries.size(); row++) {
    const Entry &entry = entries[row];
    size_t prefix = 0;
    while (prefix < previous_key.size() && prefix < entry.key.size() &&
           previous_key[prefix] == entry.key[prefix])
      prefix++;
    key_stream.push_back((uint8_t)prefix);
    key_stream.insert(key_stream.end(), entry.key.begin() + (ptrdiff_t)prefix, entry.key.end());
    previous_key = entry.key;
    tenuki.push_back(entry.tenuki);
    cell_count += entry.cells.size();
    for (const auto &cell : entry.cells)
      pair_entries[cell.first].emplace_back(row, cell.second);
  }
  std::vector<Pair> pairs;
  pairs.reserve(pair_entries.size());
  for (const auto &item : pair_entries)
    pairs.push_back(item.first);
  require(pairs.size() <= 0xffff, "too many pattern coordinates");
  Bytes geometry{kSingletonDistance, kNearDistance, kWideDistance};
  std::vector<std::pair<size_t, uint16_t>> stabilizers;
  for (size_t row = 0; row < entries.size(); row++) {
    uint16_t mask = stabilizer_mask(entries[row].key);
    if (mask)
      stabilizers.emplace_back(row, mask);
  }
  append_u16(geometry, (int)stabilizers.size());
  size_t previous_row = 0;
  for (const auto &record : stabilizers) {
    append_uvarint(geometry, record.first + 1 - previous_row);
    append_u16(geometry, record.second);
    previous_row = record.first + 1;
  }

  std::vector<int> row_means;
  for (const Entry &entry : entries) {
    int64_t sum = 0;
    for (const auto &cell : entry.cells)
      sum += cell.second;
    row_means.push_back(entry.cells.empty() ? 0 : round_ratio_half_even(sum, entry.cells.size()));
  }
  std::vector<int> row_residuals;
  int magnitude_bits = 1;
  for (size_t i = 0; i < entries.size(); i++) {
    int prediction =
        round_ratio_half_even(kRowPredictorNumerator * (int64_t)entries[i].tenuki,
                              kRowPredictorDenominator) +
        kRowPredictorIntercept;
    int residual = row_means[i] - prediction;
    row_residuals.push_back(residual);
    magnitude_bits = std::max(magnitude_bits, bit_length(std::abs(residual)));
  }
  std::vector<int> row_codes;
  for (int residual : row_residuals)
    row_codes.push_back(std::abs(residual) | ((residual < 0 ? 1 : 0) << magnitude_bits));

  struct Prediction {
    int value;
    int estimate;
    int distance;
  };
  Bytes predictor_biases;
  std::array<std::vector<int>, kSecondDistanceContexts> second_samples;
  std::vector<Prediction> predictions;
  predictions.reserve(cell_count);
  for (const Pair pair : pairs) {
    const auto &values = pair_entries.at(pair);
    std::array<std::vector<int>, kDistanceContexts> samples;
    std::vector<int> contexts;
    contexts.reserve(values.size());
    for (const auto &[row, value] : values) {
      const int context = distance_context(entries[row].key, pair);
      contexts.push_back(context);
      if (value <= 1000)
        samples[context / kDistanceContexts].push_back(value - row_means[row]);
    }
    // Center the scores on the median residual in each pair of distance classes.
    std::array<int, kDistanceContexts> bias{};
    for (size_t i = 0; i < samples.size(); i++) {
      const int quantized = quantized_median(samples[i]);
      predictor_biases.push_back((uint8_t)quantized);
      bias[i] = quantized * kBiasScale;
    }
    for (size_t i = 0; i < values.size(); i++) {
      const auto [row, value] = values[i];
      const int context = contexts[i];
      const int estimate = row_means[row] + bias[context / kDistanceContexts];
      predictions.push_back({value, estimate, context});
      if (value < 999)
        second_samples[context].push_back(value - estimate);
    }
  }
  // A shared correction incorporates the second-nearest stone of each color.
  std::array<int, kSecondDistanceContexts> second_bias{};
  for (size_t i = 0; i < second_samples.size(); i++) {
    const int quantized = quantized_median(second_samples[i]);
    predictor_biases.push_back((uint8_t)quantized);
    second_bias[i] = quantized * kBiasScale;
  }
  std::vector<int> residual_codes;
  residual_codes.reserve(cell_count);
  for (const auto &[value, estimate, distance] : predictions)
    residual_codes.push_back(stone_fraction_code(value, estimate + second_bias[distance]));
  Bytes out{'H', 'P', 'B'};
  append_u32(out, entries.size());
  append_u32(out, corpus_count);
  append_u32(out, cell_count);
  append_u32(out, key_stream.size());
  append(out, pack_bitplanes(tenuki, kFractionBits));
  append_u16(out, pairs.size());
  for (const Pair pair : pairs) {
    out.push_back((uint8_t)(int8_t)pair.first);
    out.push_back((uint8_t)(int8_t)pair.second);
  }
  append_u32(out, geometry.size());
  append(out, geometry);
  out.push_back(kRowPredictorNumerator);
  out.push_back(kRowPredictorDenominator);
  append_u16(out, kRowPredictorIntercept);
  out.push_back((uint8_t)magnitude_bits);
  append(out, pack_bitplanes(row_codes, magnitude_bits + 1));
  append(out, predictor_biases);
  append(out, pack_bitplanes(residual_codes, kFractionBits));
  append(out, key_stream);
  return out;
}

int main() {
  try {
    uint32_t corpus_count;
    const auto entries = read_entries(corpus_count);
    Bytes output = encode(entries, corpus_count);
    std::cout.write((const char *)output.data(), (std::streamsize)output.size());
    return std::cout.good() ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
