#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
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
constexpr uint16_t kBundleVersion = 1;
constexpr uint16_t kFractionMode = 3;

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
  std::map<Pair, int> cells;
};

static std::vector<Entry> read_entries() {
  Bytes input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  Reader reader(std::move(input));
  require(reader.take(4) == Bytes({'H', 'P', 'I', '1'}), "bad encoder input magic");
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
    for (uint16_t j = 0; j < cell_count; j++) {
      Pair pair{reader.i8(), reader.i8()};
      int value = reader.u16();
      require(value < kFractionLimit, "stone fraction outside u10 range");
      require(entry.cells.emplace(pair, value).second, "duplicate pattern coordinate");
    }
    entries.push_back(std::move(entry));
  }
  require(reader.done(), "trailing encoder input");
  return entries;
}

static Bytes encode(const std::vector<Entry> &entries) {
  Bytes key_stream;
  Bytes previous_key;
  std::vector<int> tenuki;
  std::set<Pair> pair_set;
  size_t cell_count = 0;
  for (const Entry &entry : entries) {
    size_t prefix = 0;
    while (prefix < previous_key.size() && prefix < entry.key.size() &&
           previous_key[prefix] == entry.key[prefix])
      prefix++;
    append_uvarint(key_stream, prefix);
    append_uvarint(key_stream, entry.key.size() - prefix);
    key_stream.insert(key_stream.end(), entry.key.begin() + (ptrdiff_t)prefix, entry.key.end());
    previous_key = entry.key;
    tenuki.push_back(entry.tenuki);
    cell_count += entry.cells.size();
    for (const auto &cell : entry.cells)
      pair_set.insert(cell.first);
  }
  std::vector<Pair> pairs(pair_set.begin(), pair_set.end());
  require(pairs.size() <= 0xffff, "too many pattern coordinates");
  std::map<Pair, size_t> pair_index;
  for (size_t i = 0; i < pairs.size(); i++)
    pair_index[pairs[i]] = i;

  size_t mask_bytes = (pairs.size() + 7) / 8;
  size_t flag_bytes = (mask_bytes + 7) / 8;
  std::vector<uint8_t> previous_presence(pairs.size(), 0);
  Bytes presence;
  for (const Entry &entry : entries) {
    std::vector<uint8_t> current(pairs.size(), 0);
    for (const auto &cell : entry.cells)
      current[pair_index.at(cell.first)] = 1;
    Bytes changed(mask_bytes, 0);
    for (size_t i = 0; i < pairs.size(); i++)
      if (previous_presence[i] != current[i])
        changed[i / 8] |= (uint8_t)(1U << (i % 8));
    Bytes flags(flag_bytes, 0);
    Bytes values;
    for (size_t i = 0; i < changed.size(); i++) {
      if (changed[i]) {
        flags[i / 8] |= (uint8_t)(1U << (i % 8));
        values.push_back(changed[i]);
      }
    }
    append(presence, flags);
    append(presence, values);
    previous_presence = std::move(current);
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
    int count = 0;
    for (const Entry &entry : entries) {
      auto found = entry.cells.find(pair);
      if (found != entry.cells.end()) {
        sum += found->second;
        count++;
      }
    }
    pair_means[pair] = round_ratio_half_even(sum, count);
  }
  std::vector<int> residual_codes;
  for (const Pair pair : pairs) {
    for (size_t i = 0; i < entries.size(); i++) {
      auto found = entries[i].cells.find(pair);
      if (found == entries[i].cells.end())
        continue;
      int residual = found->second - (row_means[i] + pair_means[pair] - global_mean);
      residual_codes.push_back(residual >= 0 ? residual << 1 : ((-residual << 1) - 1));
    }
  }
  int residual_bits = 1;
  for (int value : residual_codes)
    residual_bits = std::max(residual_bits, bit_length(value));

  Bytes out{'H', 'P', 'B', '1'};
  append_u16(out, kBundleVersion);
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
  append_u32(out, presence.size());
  append(out, presence);
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
  out.push_back((uint8_t)residual_bits);
  append(out, pack_bitplanes(residual_codes, residual_bits));
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
