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
constexpr uint16_t kFractionMode = 3;
constexpr int kResidualRiceBits = 7;
constexpr int kSingletonDistance = 16;
constexpr int kNearDistance = 7;
constexpr int kWideDistance = 4;
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

static void append_u16(Bytes &out, int value) {
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
      if ((value >> bit) & 1)
        out[offset / 8] |= (uint8_t)(1U << (offset % 8));
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

static uint16_t stabilizer_mask(const Bytes &key) {
  size_t plus_count = key[0] & 15;
  std::vector<Pair> plus, minus;
  for (size_t i = 1; i < key.size(); i++) {
    Pair point{(key[i] & 15) - 8, (key[i] >> 4) - 8};
    (i <= plus_count ? plus : minus).push_back(point);
  }
  uint16_t mask = 0;
  for (size_t index = 1; index < kTransforms.size(); index++) {
    const auto &t = kTransforms[index];
    std::vector<Pair> transformed_plus, transformed_minus;
    Pair anchor{100, 100};
    for (size_t i = 1; i < key.size(); i++) {
      int q = (key[i] & 15) - 8, r = (key[i] >> 4) - 8;
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

static Bytes pack_rice(const std::vector<int> &values, int bits) {
  std::vector<int> remainders;
  size_t unary_bits = 0;
  for (int value : values) {
    remainders.push_back(value & ((1 << bits) - 1));
    unary_bits += (value >> bits) + 1;
  }
  Bytes out = pack_bitplanes(remainders, bits);
  Bytes unary((unary_bits + 7) / 8, 0);
  size_t position = 0;
  for (int value : values) {
    for (int quotient = value >> bits; quotient > 0; quotient--, position++)
      unary[position / 8] |= (uint8_t)(1U << (position % 8));
    position++;
  }
  append(out, unary);
  return out;
}

static std::vector<Entry> read_entries() {
  Bytes input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  Reader reader(std::move(input));
  require(reader.take(3) == Bytes({'H', 'P', 'I'}), "bad encoder input magic");
  uint32_t count = reader.u32();
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
  return entries;
}

static Bytes encode(const std::vector<Entry> &entries) {
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

  int64_t all_sum = 0;
  for (const Entry &entry : entries)
    for (const auto &cell : entry.cells)
      all_sum += cell.second;
  int global_mean = cell_count == 0 ? 0 : round_ratio_half_even(all_sum, cell_count);
  std::vector<int> row_means;
  for (const Entry &entry : entries) {
    int64_t sum = 0;
    for (const auto &cell : entry.cells)
      sum += cell.second;
    row_means.push_back(entry.cells.empty() ? global_mean
                                            : round_ratio_half_even(sum, entry.cells.size()));
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

  std::map<Pair, int> pair_means;
  for (const Pair pair : pairs) {
    int64_t sum = 0;
    const auto &values = pair_entries.at(pair);
    for (const auto &value : values)
      sum += value.second;
    pair_means[pair] = round_ratio_half_even(sum, values.size());
  }
  std::vector<int> residual_codes;
  for (const Pair pair : pairs) {
    for (const auto &value : pair_entries.at(pair)) {
      int residual = value.second -
                     (row_means[value.first] + pair_means[pair] - global_mean);
      // The two marker symbols avoid predicting dead or captured cells.
      residual_codes.push_back(
          (value.second == 1001 || value.second == 1002)
              ? value.second - 1001
              : 2 + (residual >= 0 ? residual << 1 : ((-residual << 1) - 1)));
    }
  }
  Bytes out{'H', 'P', 'B'};
  append_u16(out, kFractionMode);
  append_u32(out, entries.size());
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
  std::vector<int> pair_mean_values;
  for (const Pair pair : pairs)
    pair_mean_values.push_back(pair_means[pair]);
  append(out, pack_bitplanes(pair_mean_values, kFractionBits));
  append_u16(out, global_mean);
  out.push_back((uint8_t)kResidualRiceBits);
  append(out, pack_rice(residual_codes, kResidualRiceBits));
  append(out, key_stream);
  return out;
}

int main() {
  try {
    Bytes output = encode(read_entries());
    std::cout.write((const char *)output.data(), (std::streamsize)output.size());
    return std::cout.good() ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
