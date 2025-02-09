use crate::utilities::{POINTS, TILE_COUNT};
use crate::Letter;
use std::collections::HashSet;
use trie_rs::inc_search::{Answer, IncSearch};
use trie_rs::Trie;

/// a concrete board. (0,0) is the top left corner and (3, 3) is the bottom right corner.
pub struct Board<'a> {
    /// tiles are read from left to right, top to bottom
    /// 0 1 2 3
    /// 4 5 6 7 ...
    tiles: [Letter; TILE_COUNT],
    score: u32,
    dictionary: &'a Trie<Letter>,
    // tuples are (visited_mask, position, search)
    search_stack: Vec<(u16, u8, IncSearch<'a, Letter, ()>)>,
    words: HashSet<Vec<Letter>>,
}

impl<'a> Board<'a> {
    /// full visited_mask
    const VISITED_BOARD: u16 = 0xFFFF;
    const BOARD_MIN: u8 = 0;
    const BOARD_MAX: u8 = 3;

    /// constructor
    pub fn new(tiles: [Letter; TILE_COUNT], dictionary: &'a Trie<Letter>) -> Self {
        Self {
            tiles,
            dictionary,
            score: 0,
            search_stack: Vec::new(),
            words: HashSet::new(),
        }
    }

    /// compute the maximum score for a given board
    pub fn maximum_score(&mut self) -> &u32 {
        // push the state onto the search stack (dfs for theoretical locality advantage)
        // for each starting tile
        for position in 0..TILE_COUNT {
            let visited_mask: u16 = 1 << position;
            let mut search = self.dictionary.inc_search();
            search.query(&self.tiles[position]);
            self.search_stack
                .push((visited_mask, position as u8, search));
        }

        // process all possible paths
        while let Some(cur) = self.search_stack.pop() {
            // add viable successor paths
            // this logic gets verbose, but we can avoid needlessly checking both
            // constraints on every direction at runtime by being explicit here.
            let cur_mask = cur.0;
            let position = cur.1;
            let search = cur.2;
            let (x, y) = get_xy(&position);

            // tiles to left
            if x > Self::BOARD_MIN {
                // up left
                if y > Self::BOARD_MIN {
                    self.traverse_new_path(&cur_mask, &(x - 1), &(y - 1), &search);
                }
                // level left
                self.traverse_new_path(&cur_mask, &(x - 1), &(y), &search);
                // down left
                if y < Self::BOARD_MAX {
                    self.traverse_new_path(&cur_mask, &(x - 1), &(y + 1), &search);
                }
            }
            // directly above
            if y > Self::BOARD_MIN {
                self.traverse_new_path(&cur_mask, &(x), &(y - 1), &search);
            }
            // directly below
            if y < Self::BOARD_MAX {
                self.traverse_new_path(&cur_mask, &(x), &(y + 1), &search);
            }
            // tiles to right
            if x < Self::BOARD_MAX {
                // up right
                if y > Self::BOARD_MIN {
                    self.traverse_new_path(&cur_mask, &(x + 1), &(y - 1), &search);
                }
                // level right
                self.traverse_new_path(&cur_mask, &(x + 1), &(y), &search);
                // down right
                if y < Self::BOARD_MAX {
                    self.traverse_new_path(&cur_mask, &(x + 1), &(y + 1), &search);
                }
            }
        }

        &self.score
    }

    /// attempt to create a new path from the current search path to the specified
    /// tile. does not check bounds - it is assumed that the caller has done so.
    #[inline(always)]
    fn traverse_new_path(
        &mut self,
        visited_mask: &u16,
        x: &u8,
        y: &u8,
        search: &IncSearch<'a, Letter, ()>,
    ) {
        let candidate_position = get_position(x, y);
        let position_bit = 1 << candidate_position;
        let new_mask = visited_mask | position_bit;

        // do not re-use tiles
        if new_mask == *visited_mask {
            return;
        }

        // extend the search with the new letter
        let mut new_search = search.clone();
        let postfixes_exist;
        let tile = self.tiles[candidate_position as usize].clone();
        match new_search.query(&tile) {
            Some(Answer::Prefix) => {
                postfixes_exist = true;
            }

            Some(Answer::PrefixAndMatch) => {
                postfixes_exist = true;

                self.score_word(new_search.prefix());
            }

            Some(Answer::Match) => {
                postfixes_exist = false;

                self.score_word(new_search.prefix());
            }

            None => {
                postfixes_exist = false;
            }
        }

        // push the new path onto the search stack if potentially fruitful
        if postfixes_exist && new_mask != Self::VISITED_BOARD {
            self.search_stack
                .push((new_mask, candidate_position, new_search));
        }
    }

    #[inline(always)]
    /// score points for the given word, accounting for duplicate words
    fn score_word(&mut self, word: Vec<Letter>) {
        let word_len = word.len();

        // a word can only score once per board
        if self.words.insert(word) {
            self.score += POINTS[word_len];
        }
    }
}

/// get (x, y) from position
#[inline(always)]
fn get_xy(position: &u8) -> (u8, u8) {
    let y: u8 = position / 4;
    let x: u8 = position % 4;
    (x, y)
}

/// get position from (x, y)
#[inline(always)]
fn get_position(x: &u8, y: &u8) -> u8 {
    (y * 4) + x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::letter::translate_word;
    use crate::utilities::create_trie;
    use std::collections::HashMap;

    #[test]
    fn test_maximum_score() {
        // test design is to test against a real boards (and solutions) from a high-scoring
        // game to cover a variety of traversal scenarios.
        let word_list = create_trie("wordlist-actual.txt").0;
        // F L O S
        // E I M T
        // R K E I
        // O Y R A
        const BOARD_1: &str = "FLOSEIMTRKEIOYRA";
        let tiles_1 = translate_word(BOARD_1).unwrap();
        let mut board_1 = Board::new(tiles_1.try_into().unwrap(), &word_list);

        // manually verified that:
        // - a word begins and ends (except ending at the bottom left tile) at every tile in this board
        // - board contains duplicate words
        // - board contains words of length [3, 8]
        // - board contains words using connections in all directions
        const SOLUTION_WORDS_1: [&str; 287] = [
            "SMIRKIER", "STEMLIKE", "FILMIER", "LIMITER", "MIRKIER", "OSTIARY", "REFILMS",
            "SMIRKER", "ELMIER", "ETOILE", "FIKERY", "FILMER", "KELIMS", "KRAITS", "LIMIER",
            "LIMITS", "MERITS", "MIRKER", "MOILER", "MOTIER", "OILERY", "ORIOLE", "REFILM",
            "RELIER", "REMITS", "RIMIER", "RIOTER", "ROKIER", "SMEARY", "SMILER", "SMIRKY",
            "SMITER", "SMOILE", "SOMITE", "STIMIE", "TOILER", "YORKER", "YORKIE", "AIERY", "AIMER",
            "ARETS", "EMITS", "EYRIE", "FIERY", "FILER", "FILMI", "FILMS", "FILOS", "FLIER",
            "FLOTE", "ITEMS", "KEFIR", "KELIM", "KETOL", "KILOS", "KRAIT", "KYRIE", "LIFER",
            "LIKER", "LIMEY", "LIMIT", "LIMOS", "MERIT", "METOL", "MIKRA", "MILER", "MILOS",
            "MIRKY", "MITER", "MOIRE", "MOSTE", "MOTEY", "OILER", "OMITS", "ORIEL", "OSTIA",
            "RAITS", "REKEY", "RELIE", "REMIT", "RETIA", "RIEMS", "RIFLE", "RIMER", "RIOTS",
            "ROKER", "SMEAR", "SMERK", "SMILE", "SMIRK", "SMITE", "SMOTE", "SOLEI", "SOLER",
            "STEAR", "STEIL", "STIME", "STIRE", "STIRK", "STOLE", "TEARY", "TERAI", "TIMER",
            "TOILE", "TOMIA", "YOKEL", "AERY", "AIMS", "AIRY", "AITS", "ARET", "ELMS", "EMIR",
            "EMIT", "EMOS", "EYRA", "EYRE", "FIKE", "FIKY", "FILE", "FILM", "FILO", "FIRE", "FIRK",
            "FLIR", "ITEM", "KEIR", "KETO", "KETS", "KIEF", "KIER", "KILO", "KORE", "KRAI", "LEIR",
            "LEKE", "LIEF", "LIER", "LIFE", "LIKE", "LIME", "LIMO", "LIRE", "LIRK", "LOIR", "LOME",
            "LOST", "LOTE", "LOTI", "LOTS", "MERI", "MERK", "METS", "MIKE", "MILE", "MILO", "MIRE",
            "MIRK", "MIRY", "MITE", "MOIL", "MOLE", "MOST", "MOTE", "MOTS", "OKRA", "OLMS", "OMER",
            "OMIT", "OYER", "RAIT", "REIF", "REIK", "REKE", "REMS", "RETS", "RIEL", "RIEM", "RIFE",
            "RILE", "RIME", "RIMS", "RIOT", "RITE", "RITS", "ROKE", "ROKY", "RYKE", "SMIR", "SMIT",
            "SOIL", "SOLE", "SOLI", "SOME", "STEM", "STEY", "STIE", "STIR", "TEAR", "TEIL", "TEMS",
            "TIAR", "TIER", "TIME", "TIRE", "TOIL", "TOLE", "TOME", "TOMS", "YEAR", "YERK", "YETI",
            "YOKE", "YORE", "YORK", "AIM", "AIR", "AIT", "ARE", "ARK", "ARY", "EAR", "EIK", "EKE",
            "ELF", "ELM", "EMO", "EMS", "ERA", "ERK", "FER", "FIE", "FIL", "FIR", "IOS", "IRE",
            "IRK", "ITS", "KEA", "KEF", "KET", "KEY", "KIF", "KIR", "KOR", "KYE", "LEI", "LEK",
            "LIE", "LOS", "LOT", "MET", "MIL", "MIR", "MOI", "MOL", "MOS", "MOT", "OIK", "OIL",
            "OKE", "OLE", "OLM", "OMS", "ORE", "OYE", "RAI", "REF", "REI", "REM", "RET", "RIA",
            "RIF", "RIM", "RIT", "ROK", "RYE", "SOL", "SOM", "SOT", "TEA", "TIE", "TOM", "YEA",
            "YER", "YET", "YOK",
        ];

        let expected_score_1: u32 = SOLUTION_WORDS_1
            .iter()
            .map(|word| POINTS[word.chars().count()])
            .sum();
        let expected_words_1 = HashSet::from_iter(
            SOLUTION_WORDS_1
                .iter()
                .map(|word| translate_word(word).unwrap()),
        );

        let actual_score_1 = board_1.maximum_score();

        // check that we found all words and that the score is as expected
        assert_eq!(&expected_score_1, actual_score_1);
        assert_eq!(
            expected_words_1
                .symmetric_difference(&board_1.words)
                .count(),
            0
        );

        // second board. contains a word that ends at the bottom left corner.
        // also just the more the merrier!
        // S O H I
        // E T T C
        // D A A C
        // M E N K
        const BOARD_2: &str = "SOHIETTCDAACMENK";
        let tiles_2 = translate_word(BOARD_2).unwrap();
        let mut board_2 = Board::new(tiles_2.try_into().unwrap(), &word_list);

        // manually verified that:
        // - a word begins and ends (except ending at the bottom left tile) at every tile in this board
        // - board contains duplicate words
        // - board contains words of length [3, 8]
        // - board contains words using connections in all directions
        const SOLUTION_WORDS_2: [&str; 252] = [
            "EMANATED", "EMANATES", "OEDEMATA", "ANATTOS", "ATACTIC", "CHITTED", "EDEMATA",
            "EMANATE", "NEMATIC", "STEAMED", "STEANED", "ANADEM", "ANATTO", "ANEATH", "ATTACH",
            "ATTACK", "CATCHT", "CATTED", "CHOTTS", "DATTOS", "DETACH", "ENATES", "ENATIC",
            "HOSTED", "HOTTED", "MANATI", "MANATS", "MATTED", "MATTES", "MEATED", "NACHOS",
            "OEDEMA", "SEAMED", "SEAMEN", "SEANED", "SOTTED", "STAMEN", "STANCK", "STANED",
            "STATIC", "STEANE", "TACHOS", "TACTIC", "TEAMED", "ACNED", "ADMEN", "AEDES", "ATTIC",
            "CACTI", "CANED", "CATCH", "CATES", "CHOSE", "CHOTA", "CHOTT", "DATES", "DATOS",
            "DATTO", "DEATH", "DEMAN", "EANED", "EDEMA", "ENACT", "ENATE", "ETHIC", "ETHOS",
            "HOSED", "HOSTA", "KATTI", "KNEAD", "MAAED", "MANAT", "MANED", "MATCH", "MATED",
            "MATES", "MATTE", "MATTS", "MEATH", "MEATS", "MENAD", "NACHO", "NAMED", "NATCH",
            "NATES", "NEATH", "NEATS", "SEAME", "SEDAN", "SETAE", "STACK", "STADE", "STANE",
            "STANK", "STEAD", "STEAM", "STEAN", "STEDE", "TACHO", "TACIT", "TAMED", "TANKA",
            "TATES", "TEADE", "TEAED", "THOSE", "TICCA", "TOSED", "TOTED", "TOTES", "ACNE", "ACTA",
            "AMEN", "ATES", "CANE", "CATE", "CATS", "CHIT", "CITO", "DAES", "DAME", "DANK", "DATA",
            "DATE", "DATO", "DEAN", "EACH", "EATH", "EATS", "ETAT", "HOED", "HOES", "HOSE", "HOST",
            "HOTE", "HOTS", "ITCH", "KAED", "KANA", "KANE", "KATA", "KATI", "KATS", "MADE", "MAES",
            "MANA", "MANE", "MATE", "MATH", "MATS", "MATT", "MEAD", "MEAN", "MEAT", "NAAM", "NACH",
            "NAES", "NAME", "NATS", "NEAT", "NEMA", "OTIC", "SEAM", "SEAN", "SEAT", "SETA", "SETT",
            "SOTH", "STAT", "STED", "STOT", "TACH", "TACK", "TACT", "TAED", "TAES", "TAME", "TANA",
            "TANE", "TANK", "TATE", "TATH", "TATS", "TEAD", "TEAM", "TEAT", "TICH", "TOEA", "TOED",
            "TOES", "TOSE", "TOST", "TOTE", "TOTS", "ACH", "ACT", "ANA", "ANE", "ATE", "ATT",
            "CAN", "CAT", "CHI", "CIT", "DAE", "DAM", "DAN", "DEN", "EAN", "EAT", "EST", "ETA",
            "ETH", "HIC", "HIT", "HOE", "HOT", "ICH", "ITA", "KAE", "KAT", "MAA", "MAD", "MAE",
            "MAN", "MAT", "MED", "MEN", "NAE", "NAM", "NAT", "NED", "OES", "OSE", "SEA", "SED",
            "SET", "SOH", "SOT", "TAD", "TAE", "TAK", "TAM", "TAN", "TAT", "TEA", "TED", "TES",
            "THO", "TIC", "TOE", "TOT",
        ];

        let expected_score_2: u32 = SOLUTION_WORDS_2
            .iter()
            .map(|word| POINTS[word.chars().count()])
            .sum();
        let expected_words_2 = HashSet::from_iter(
            SOLUTION_WORDS_2
                .iter()
                .map(|word| translate_word(word).unwrap()),
        );

        let actual_score_2 = board_2.maximum_score();

        // check that we found all words and that the score is as expected
        assert_eq!(&expected_score_2, actual_score_2);
        assert_eq!(
            expected_words_2
                .symmetric_difference(&board_2.words)
                .count(),
            0
        );

        // test words of length [9, 15]

        // A A R D
        // K R A V
        // S Z Z Z
        // Z Z Z Z
        const BOARD_3: &str = "AARDKRAVSZZZZZZZ";
        let tiles_3 = translate_word(BOARD_3).unwrap();
        let mut board_3 = Board::new(tiles_3.try_into().unwrap(), &word_list);
        board_3.maximum_score();
        // length 9
        assert!(board_3
            .words
            .contains(&translate_word("AARDVARKS").unwrap()));

        // A C R O
        // I T A B
        // S M X X
        // X X X X
        const BOARD_4: &str = "ACROITABSMXXXXXX";
        let tiles_4 = translate_word(BOARD_4).unwrap();
        let mut board_4 = Board::new(tiles_4.try_into().unwrap(), &word_list);
        board_4.maximum_score();
        // length 10
        assert!(board_4
            .words
            .contains(&translate_word("ACROBATISM").unwrap()));

        // A E T Z
        // B N I Y
        // I E C L
        // O G A L
        const BOARD_5: &str = "AETZBNIYIECLOGAL";
        let tiles_5 = translate_word(BOARD_5).unwrap();
        let mut board_5 = Board::new(tiles_5.try_into().unwrap(), &word_list);
        board_5.maximum_score();
        // lengths 11, 15
        assert!(board_5
            .words
            .contains(&translate_word("ABIOGENETIC").unwrap()));
        assert!(board_5
            .words
            .contains(&translate_word("ABIOGENETICALLY").unwrap()));

        // C A P I
        // A L U T
        // R I E S
        // B B B B
        const BOARD_6: &str = "CAPIALUTRIESBBBB";
        let tiles_6 = translate_word(BOARD_6).unwrap();
        let mut board_6 = Board::new(tiles_6.try_into().unwrap(), &word_list);
        board_6.maximum_score();
        // length 12
        assert!(board_6
            .words
            .contains(&translate_word("CAPITULARIES").unwrap()));

        // Y E E E
        // L S U O
        // D I T I
        // E P X E
        const BOARD_7: &str = "YEEELSUODITIEPXE";
        let tiles_7 = translate_word(BOARD_7).unwrap();
        let mut board_7 = Board::new(tiles_7.try_into().unwrap(), &word_list);
        board_7.maximum_score();
        // length 13
        assert!(board_7
            .words
            .contains(&translate_word("EXPEDITIOUSLY").unwrap()));

        // C H I N
        // R E H C
        // I N C H
        // X X E E
        const BOARD_8: &str = "CHINREHCINCHXXEE";
        let tiles_8 = translate_word(BOARD_8).unwrap();
        let mut board_8 = Board::new(tiles_8.try_into().unwrap(), &word_list);
        board_8.maximum_score();
        // length 14
        assert!(board_8
            .words
            .contains(&translate_word("CHINCHERINCHEE").unwrap()));
    }
}
