use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fmt;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
/// simplify storage as our grammar consists of only upper case english letters
/// as opposed to all unicode values.
pub enum Letter {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
    F = 5,
    G = 6,
    H = 7,
    I = 8,
    J = 9,
    K = 10,
    L = 11,
    M = 12,
    N = 13,
    O = 14,
    P = 15,
    Q = 16,
    R = 17,
    S = 18,
    T = 19,
    U = 20,
    V = 21,
    W = 22,
    X = 23,
    Y = 24,
    Z = 25,
}

impl Into<usize> for Letter {
    fn into(self) -> usize {
        self as usize
    }
}

impl From<usize> for Letter {
    fn from(n: usize) -> Letter {
        match n {
            0 => Letter::A,
            1 => Letter::B,
            2 => Letter::C,
            3 => Letter::D,
            4 => Letter::E,
            5 => Letter::F,
            6 => Letter::G,
            7 => Letter::H,
            8 => Letter::I,
            9 => Letter::J,
            10 => Letter::K,
            11 => Letter::L,
            12 => Letter::M,
            13 => Letter::N,
            14 => Letter::O,
            15 => Letter::P,
            16 => Letter::Q,
            17 => Letter::R,
            18 => Letter::S,
            19 => Letter::T,
            20 => Letter::U,
            21 => Letter::V,
            22 => Letter::W,
            23 => Letter::X,
            24 => Letter::Y,
            25 => Letter::Z,
            _ => panic!("Cannot convert values greater than 25 into a letter."),
        }
    }
}

impl fmt::Display for Letter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", letter_to_char(self))
    }
}

/// marker type for letter-specific implementation of trie-rs library function
/*#[derive(Clone)]
pub struct LetterCollect;

impl TryFromIterator<Letter, LetterCollect> for Vec<Letter> {
    type Error = ();
    fn try_from_iter<T>(iter: T) -> Result<Self, Self::Error>
    where
    Self: Sized,
    T: IntoIterator<Item = Letter>,
    {
    Ok(iter.into_iter().collect::<Vec<Letter>>())
    }
} */

static TRANSLATOR: Lazy<Translator> = Lazy::new(Translator::new);

pub fn translate_letter(c: &char) -> Option<Letter> {
    TRANSLATOR.translate_letter(c)
}

pub fn translate_word(word: &str) -> Result<Vec<Letter>, &'static str> {
    TRANSLATOR.translate_word(word)
}

pub fn to_word(letters: &[Letter]) -> String {
    TRANSLATOR.to_word(letters)
}

pub fn letter_to_char(letter: &Letter) -> char {
    TRANSLATOR.letter_to_char(letter)
}

pub struct Translator {
    letter_map: HashMap<char, Letter>,
    char_map: HashMap<Letter, char>,
}

impl Translator {
    pub fn new() -> Self {
        Self {
            letter_map: HashMap::from([
                ('A', Letter::A),
                ('B', Letter::B),
                ('C', Letter::C),
                ('D', Letter::D),
                ('E', Letter::E),
                ('F', Letter::F),
                ('G', Letter::G),
                ('H', Letter::H),
                ('I', Letter::I),
                ('J', Letter::J),
                ('K', Letter::K),
                ('L', Letter::L),
                ('M', Letter::M),
                ('N', Letter::N),
                ('O', Letter::O),
                ('P', Letter::P),
                ('Q', Letter::Q),
                ('R', Letter::R),
                ('S', Letter::S),
                ('T', Letter::T),
                ('U', Letter::U),
                ('V', Letter::V),
                ('W', Letter::W),
                ('X', Letter::X),
                ('Y', Letter::Y),
                ('Z', Letter::Z),
            ]),
            char_map: HashMap::from([
                (Letter::A, 'A'),
                (Letter::B, 'B'),
                (Letter::C, 'C'),
                (Letter::D, 'D'),
                (Letter::E, 'E'),
                (Letter::F, 'F'),
                (Letter::G, 'G'),
                (Letter::H, 'H'),
                (Letter::I, 'I'),
                (Letter::J, 'J'),
                (Letter::K, 'K'),
                (Letter::L, 'L'),
                (Letter::M, 'M'),
                (Letter::N, 'N'),
                (Letter::O, 'O'),
                (Letter::P, 'P'),
                (Letter::Q, 'Q'),
                (Letter::R, 'R'),
                (Letter::S, 'S'),
                (Letter::T, 'T'),
                (Letter::U, 'U'),
                (Letter::V, 'V'),
                (Letter::W, 'W'),
                (Letter::X, 'X'),
                (Letter::Y, 'Y'),
                (Letter::Z, 'Z'),
            ]),
        }
    }

    fn translate_letter(&self, c: &char) -> Option<Letter> {
        self.letter_map.get(c).cloned()
    }

    fn translate_word(&self, word: &str) -> Result<Vec<Letter>, &'static str> {
        let word_as_letters: Vec<Option<Letter>> = word
            .chars()
            .map(|c| self.letter_map.get(&c).cloned())
            .collect();
        if word_as_letters.iter().any(|l| l.is_none()) {
            Err("Could not decode character.")
        } else {
            Ok(word_as_letters.into_iter().flatten().collect())
        }
    }

    fn to_word(&self, letters: &[Letter]) -> String {
        letters.iter().map(|l| self.char_map[l]).collect()
    }

    fn letter_to_char(&self, letter: &Letter) -> char {
        match self.char_map.get(letter) {
            Some(c) => *c,
            None => '!',
        }
    }
}
