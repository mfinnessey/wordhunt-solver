use crate::utilities::{POINTS, TILE_COUNT};
use crate::Letter;
use std::collections::HashSet;
use trie_rs::inc_search::{Answer, IncSearch};
use trie_rs::Trie;

/// a concrete board. (0,0) is the top left corner and (3, 3) is the bottom right corner.
pub struct Board<'a> {
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
    // TODO unit test this - correctness issue - use scores from official word hunts
    pub fn maximum_score(&mut self) -> &u32 {
        // push the state onto the search stack (dfs for theoretical locality advantage)
        // for each starting tile
        for position in 0..15 {
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
            Some(Answer::PrefixAndMatch) | Some(Answer::Match) => {
                postfixes_exist = true;

                // only valid words of length at least 3 score
                if new_search.prefix_len() >= 3 {
                    self.score += POINTS[new_search.prefix_len()];
                    self.words.insert(new_search.prefix());
                }
            }

            Some(Answer::Prefix) => {
                postfixes_exist = true;
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

    pub fn get_words(&self) -> Vec<&Vec<Letter>> {
        self.words.iter().collect()
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
