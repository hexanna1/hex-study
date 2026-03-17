#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using Point = std::pair<int, int>;

struct Position {
  int size;
  bool red_to_play;
  std::vector<Point> red;
  std::vector<Point> blue;
  std::unordered_set<int> occupied;
};

static int key(Point p, int size) { return p.first * (size + 1) + p.second; }

static Point parse_cell(const std::string &s) {
  size_t i = 0;
  int col = 0;
  while (i < s.size() && s[i] >= 'a' && s[i] <= 'z') {
    col = col * 26 + (s[i] - 'a' + 1);
    i++;
  }
  if (i == 0 || i == s.size())
    throw std::runtime_error("bad cell");
  for (size_t j = i; j < s.size(); j++)
    if (s[j] < '0' || s[j] > '9')
      throw std::runtime_error("bad cell");
  int row = std::stoi(s.substr(i));
  return {col, row};
}

static std::string cell(Point p) {
  int col = p.first;
  std::string letters;
  while (col > 0) {
    col--;
    letters.push_back(char('a' + col % 26));
    col /= 26;
  }
  std::reverse(letters.begin(), letters.end());
  return letters + std::to_string(p.second);
}

static Position parse_position(const std::string &input) {
  size_t hash = input.find('#');
  std::string frag = hash == std::string::npos ? input : input.substr(hash + 1);
  size_t marker = frag.find("c1,");
  if (marker == std::string::npos || marker == 0)
    throw std::runtime_error("bad position");
  Position out;
  out.size = std::stoi(frag.substr(0, marker));
  std::string stream = frag.substr(marker + 3);
  bool red = true;
  for (size_t i = 0; i < stream.size();) {
    if (stream.compare(i, 2, ":p") == 0) {
      i += 2;
    } else {
      size_t start = i;
      while (i < stream.size() && stream[i] >= 'a' && stream[i] <= 'z')
        i++;
      while (i < stream.size() && stream[i] >= '0' && stream[i] <= '9')
        i++;
      Point p = parse_cell(stream.substr(start, i - start));
      (red ? out.red : out.blue).push_back(p);
      out.occupied.insert(key(p, out.size));
    }
    red = !red;
  }
  out.red_to_play = red;
  return out;
}

static Point transform(Point p, bool swap) {
  return swap ? Point{p.second, p.first} : p;
}
static Point place(Point p, int size, bool top_right) {
  return top_right ? Point{size + 1 - p.second, size + 1 - p.first}
                   : Point{p.second, p.first};
}

static std::vector<Point> variant(const std::vector<Point> &src, int size,
                                  bool swap, bool top_right) {
  std::vector<Point> out;
  for (Point p : src)
    out.push_back(place(transform(p, swap), size, top_right));
  return out;
}

static bool subset(const std::vector<Point> &points,
                   const std::unordered_set<int> &stones, int size) {
  for (Point p : points)
    if (!stones.count(key(p, size)))
      return false;
  return true;
}
static bool disjoint(const std::vector<Point> &points,
                     const std::unordered_set<int> &occupied, int size) {
  for (Point p : points)
    if (occupied.count(key(p, size)))
      return false;
  return true;
}

struct AcuteContext {
  std::unordered_set<int> dead;
  std::unordered_map<int, Point> canonical_moves;
};

static AcuteContext acute_context(const Position &pos) {
  const std::vector<Point> dead_red_anchors{{4, 2}};
  const std::vector<Point> dead_blue_anchors{{2, 3}};
  const std::vector<Point> dead_region{{1, 1}, {2, 1}, {3, 1}, {4, 1},
                                       {1, 2}, {2, 2}, {3, 2}};
  const std::vector<Point> equivalence_red_anchors{{4, 2}, {4, 3}};
  const std::vector<Point> equivalence_blue_anchors{{3, 3}};
  const std::vector<Point> equivalence_required_empty{
      {1, 1}, {2, 1}, {3, 1}, {4, 1}, {1, 2}, {2, 2}, {3, 2}, {1, 3}, {2, 3}};
  std::unordered_set<int> redset, blueset;
  for (Point p : pos.red)
    redset.insert(key(p, pos.size));
  for (Point p : pos.blue)
    blueset.insert(key(p, pos.size));
  AcuteContext out;
  for (bool swap : {false, true})
    for (bool top_right : {false, true}) {
      auto rule_red = variant(swap ? dead_blue_anchors : dead_red_anchors,
                              pos.size, swap, top_right);
      auto rule_blue = variant(swap ? dead_red_anchors : dead_blue_anchors,
                               pos.size, swap, top_right);
      auto rule_dead = variant(dead_region, pos.size, swap, top_right);
      if (subset(rule_red, redset, pos.size) &&
          subset(rule_blue, blueset, pos.size) &&
          disjoint(rule_dead, pos.occupied, pos.size))
        for (Point p : rule_dead)
          out.dead.insert(key(p, pos.size));

      rule_red =
          variant(swap ? equivalence_blue_anchors : equivalence_red_anchors,
                  pos.size, swap, top_right);
      rule_blue =
          variant(swap ? equivalence_red_anchors : equivalence_blue_anchors,
                  pos.size, swap, top_right);
      auto rule_empty =
          variant(equivalence_required_empty, pos.size, swap, top_right);
      if (subset(rule_red, redset, pos.size) &&
          subset(rule_blue, blueset, pos.size) &&
          disjoint(rule_empty, pos.occupied, pos.size)) {
        Point loser = place(transform({2, 3}, swap), pos.size, top_right);
        Point winner = place(transform({3, 2}, swap), pos.size, top_right);
        out.canonical_moves[key(loser, pos.size)] = winner;
      }
    }
  return out;
}

static std::string serialize_child(const Position &pos, Point move) {
  std::vector<Point> red = pos.red, blue = pos.blue;
  (pos.red_to_play ? red : blue).push_back(move);
  std::sort(red.begin(), red.end());
  std::sort(blue.begin(), blue.end());
  size_t ri = 0, bi = 0;
  bool side_red = true;
  std::string stream;
  while (ri < red.size() || bi < blue.size()) {
    if (side_red)
      stream += ri < red.size() ? cell(red[ri++]) : ":p";
    else
      stream += bi < blue.size() ? cell(blue[bi++]) : ":p";
    side_red = !side_red;
  }
  bool next_red = !pos.red_to_play;
  if (side_red != next_red)
    stream += ":p";
  return "https://hexworld.org/board/#" + std::to_string(pos.size) + "c1," +
         stream;
}

static std::vector<std::string> split(const std::string &value,
                                      char delimiter) {
  std::vector<std::string> out;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, delimiter))
    out.push_back(item);
  return out;
}

int main() {
  try {
    std::string line;
    if (!std::getline(std::cin, line))
      throw std::runtime_error("missing configuration");
    auto config = split(line, '\t');
    if (config.size() == 2 && config[0] == "children-v1") {
      int board_size = std::stoi(config[1]);
      while (std::getline(std::cin, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos)
          throw std::runtime_error("bad child request");
        Position pos = parse_position(line.substr(0, tab));
        if (pos.size != board_size)
          throw std::runtime_error("board-size mismatch");
        auto moves = split(line.substr(tab + 1), ';');
        bool first = true;
        for (const std::string &move_text : moves) {
          if (move_text.empty())
            continue;
          Point move = parse_cell(move_text);
          if (pos.occupied.count(key(move, pos.size)))
            throw std::runtime_error("child move already occupied");
          if (!first)
            std::cout << '\t';
          first = false;
          std::cout << serialize_child(pos, move);
        }
        std::cout << '\n';
      }
      return 0;
    }
    if (config.size() != 12 || config[0] != "opening-v1")
      throw std::runtime_error("bad configuration");
    int ply = std::stoi(config[1]);
    int board_size = std::stoi(config[2]);
    int top_k = std::stoi(config[3]);
    double importance_min = std::stod(config[4]);
    double ply_decay = std::stod(config[5]);
    double extra_prior_min = std::stod(config[6]);
    double prior_log_step = std::stod(config[7]);
    double rank_step = std::stod(config[8]);
    double ply_step = std::stod(config[9]);
    double importance_headroom = std::stod(config[10]);
    int top_k_headroom = std::stoi(config[11]);
    while (std::getline(std::cin, line)) {
      size_t a = line.find('\t'), b = line.find('\t', a + 1);
      if (a == std::string::npos || b == std::string::npos)
        throw std::runtime_error("bad opening request");
      Position pos = parse_position(line.substr(0, a));
      if (pos.size != board_size)
        throw std::runtime_error("board-size mismatch");
      double importance = std::stod(line.substr(a + 1, b - a - 1));
      AcuteContext acute = acute_context(pos);
      std::unordered_set<int> seen;
      std::stringstream policies(line.substr(b + 1));
      std::string item;
      int seq = 0, cleaned = 0;
      int proof_raw_rows = 0, proof_cleaned_rank = 0;
      bool first = true;
      while (std::getline(policies, item, ';')) {
        if (item.empty())
          continue;
        size_t comma = item.find(',');
        if (comma == std::string::npos)
          throw std::runtime_error("bad policy row");
        std::string move_s = item.substr(0, comma);
        std::string prior_encoded = item.substr(comma + 1);
        double prior = std::stod(prior_encoded) / 1000000.0;
        seq++;
        if (move_s == "pass")
          continue;
        Point move = parse_cell(move_s);
        int move_key = key(move, pos.size);
        if (pos.occupied.count(move_key) || acute.dead.count(move_key))
          continue;
        auto mapped = acute.canonical_moves.find(move_key);
        if (mapped != acute.canonical_moves.end())
          move = mapped->second;
        move_key = key(move, pos.size);
        if (seen.count(move_key))
          continue;
        seen.insert(move_key);
        cleaned++;
        auto candidate_weight = [&](int effective_top_k) {
          if (cleaned <= effective_top_k || prior >= extra_prior_min)
            return 1.0;
          double prior_log = -std::log10(std::max(1e-6, prior));
          double exponent = prior_log_step * prior_log +
                            rank_step * std::max(0, cleaned - effective_top_k - 1) +
                            ply_step * std::max(0, ply - 1);
          return std::pow(importance_min, exponent);
        };
        double weight = candidate_weight(top_k);
        double proof_weight = candidate_weight(top_k + top_k_headroom);
        bool proof_keep = importance * ply_decay * proof_weight *
                              importance_headroom >=
                          importance_min;
        if (!proof_keep) {
          if (proof_raw_rows == 0) {
            proof_raw_rows = seq;
            proof_cleaned_rank = cleaned;
          }
        } else {
          proof_raw_rows = 0;
          proof_cleaned_rank = 0;
        }
        bool keep = importance * ply_decay * weight >= importance_min;
        if (!keep)
          continue;
        std::string child = serialize_child(pos, move);
        if (!first)
          std::cout << '\t';
        first = false;
        std::cout << cell(move) << '|' << seq << '|' << prior_encoded << '|'
                  << cleaned << '|' << (pos.red_to_play ? "red" : "blue") << '|'
                  << child;
      }
      if (!first)
        std::cout << '\t';
      std::cout << "@|" << proof_raw_rows << '|' << proof_cleaned_rank;
      std::cout << '\n';
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
