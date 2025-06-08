# Purpose
Find the 4 x 4 (Map 1) WordHunt board with the largest possible maximum score.

# Problem Outline
A 4 x 4 WordHunt board consists of 16 (English) alphabetic tiles arranged in a grid with 4 rows and 4 columns. I am going to ignore any (potential) board generation constraints and consider all boards that could exist subject to the obvious constraints. Paths are created by connecting adjacent (vertically, horizontally, or diagonally) tiles in a path such that no tile is used more than once. This path will then score points if and only if it forms a valid word that has not been previously scored. Words are scored proportional to their length, with longer words scoring more points according to a piece-wise function that is laid out below.

|Length (Characters)|Score (Points)|
|-------------------|--------------|
|0|0|
|1|0|
|2|0|
|3|100|
|4|400|
|5|800|
|6|1400|
|7|1800|
|8|2200|
|9|2600|
|10|3000|
|11|3400|
|12|3800|
|13|4200|
|14|4600|
|15|5000|
|16|5400|


# Approach
A naive, brute force algorithm would look something like this
```
FindMaxBoard(WordList)
	Set MaxScore = 0
	For Each Board In GenerateBoards()
		Set FoundWords = {}
		Set BoardScore = 0
		For Each Tile in Board
			Set (BoardScore, FoundWords) = (BoardScore + AnalyzeBoard(WordList, Tile, FoundWords)[0], AnalyzeBoard(WordList, Tile, FoundWords)[1])
		Set MaxScore = Max(BoardScore, MaxScore)
	
AnalyzeBoard(WordList, Tile, FoundWords)	
	Set WordsValue = 0
	For Word in SequencesFrom(Tile)
		If Word Not In FoundWords
			Set WordsValue = WordsValue + WordValueTable[Length(Word)]
			Add Word to FoundWords
	Return WordsValue, FoundWords
```

However, what this approach has in simplicity, it lacks in efficiency. The search space is simply too large. Considering that there are 16 tiles on the board and 26 options for each tile, there are $26^{16}$ or a hair over 43 sextillion (!!) possible boards. Even with some pretty big (and/or distributed) iron, I won't get the answers that I desperately seek ~~until the heat death of the universe~~ anytime soon. That's lead to a bunch of planning, testing, re-planning, and (some) optimization to see what I can do to solve this problem anyways :)

# Status
Stage 1 (letter combination) Search design is complete. It currently involves a tiled iteration over the word list using `N` (where `N` is the amount of available parallelism) worker threads, a thread to generate the letter combinations, and a thread to periodically take snapshots (the first stage 1 search is estimated to take approximately 17 days on my Ryzen 9 9950x). There are associated unit and integration tests for this.

This repository is a work-in-progress. Not all stages are complete. I anticipate that this project might take me a while to complete, and I'm okay with that. I've tried to clean up most of the code, but it's not "production-quality" by any means. It's not fully-optimized either. In the interest of actually getting started, I've left some fairly obvious optimizations on the table for now.

I'll also note that this is my first project writing Rust (this project was half started as an excuse to practice writing Rust). As I've learned more, I've endeavored to fix my earlier errors, but I'm sure that there are plenty of places in here that could benefit from significant refactoring.

# Running
You will need to provide your own word list. After that, it's a simple `cargo run --release --bin $BIN_NAME`. The available binaries are currently one for stage 1 search 1 and a binary to generate a list of random letter combinations for benchmarking or similar.
