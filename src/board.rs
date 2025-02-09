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
        const BOARD: &str = "FLOSEIMTRKEIOYRA";
        let tiles = translate_word(BOARD).unwrap();
        let mut board = Board::new(tiles.try_into().unwrap(), &word_list);

        // manually verified that:
        // - a word begins and ends (except ending at the bottom left tile) at every tile in this board
        // - board contains duplicate words
        // - board contains words of length [3, 8]
        // - board contains words using connections in all directions
        const SOLUTION_WORDS: [&str; 287] = [
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

        let expected_score: u32 = SOLUTION_WORDS
            .iter()
            .map(|word| POINTS[word.chars().count()])
            .sum();
        let expected_words = HashSet::from_iter(
            SOLUTION_WORDS
                .iter()
                .map(|word| translate_word(word).unwrap()),
        );

        let actual_score = board.maximum_score();

        // check that we found all words and that the score is as expected
        assert_eq!(&expected_score, actual_score);
        assert_eq!(expected_words.symmetric_difference(&board.words).count(), 0);

        // TODO test words of length [9, 15]
    }
}
