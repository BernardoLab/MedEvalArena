#!/usr/bin/env Rscript
# Topic difficulty only (stand-alone):
#   - Are some topics harder than others?
#   - Item is the unit; item score = mean correctness across LLMs for that item.
#   - Outputs:
#       1) topic_difficulty_bootstrap.csv  (topic means + bootstrap CIs)
#       2) topic_global_test_permutation.csv (global permutation test + effect size)
#       3) item_scores.csv (item-level p_item used for analysis)
#       4) topic_difficulty_summary.txt
#
# Example:
#   Rscript topic_difficulty_only.R \
#     --runs_root eval_results/quizbench/quizzes_ABMS20260101/runs \
#     --outdir R_results/topic_difficulty_ABMS20260101 \
#     --boot_reps 5000 \
#     --perm_reps 5000 \
#     --seed 2026
#
# Or with CSV:
#   Rscript topic_difficulty_only.R \
#     --input_csv item_level.csv \
#     --outdir R_results/topic_difficulty_from_csv \
#     --boot_reps 5000 --perm_reps 5000 --seed 2026

required_pkgs <- c("dplyr", "tidyr", "jsonlite")
missing <- required_pkgs[!vapply(required_pkgs, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing) > 0) {
  stop(
    paste0(
      "Missing required R packages: ", paste(missing, collapse = ", "), "\n",
      "Install with install.packages(c(", paste(sprintf("%s", shQuote(missing)), collapse = ", "), "))."
    )
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(jsonlite)
})

usage <- function() {
  cat(paste(
    "Topic difficulty only (paired by item across LLMs).",
    "",
    "Input options (choose one):",
    "  --runs_root PATH     QuizBench runs root containing *_result.json files",
    "  --input_csv PATH     CSV with columns: item, llm, y, topic (topic required for analysis)",
    "",
    "Optional:",
    "  --outdir PATH                 Output directory (default: R_results/topic_difficulty_<timestamp>)",
    "  --boot_reps N                 Bootstrap replicates for topic CIs + range CI (default: 5000)",
    "  --perm_reps N                 Permutation replicates for global topic test (default: 5000)",
    "  --seed N                      RNG seed (default: 2026)",
    "  --drop_incomplete_items BOOL  Drop items missing any LLM response (default: true)",
    "  --perm_unstratified BOOL      If false and quiz labels exist, permute topics within quiz strata (default: false)",
    "  --min_items_per_topic N       Minimum items required to include a topic (default: 5)",
    "  --generator_only BOOL         Keep only LLMs that also appear as quiz generators (default: false)",
    "  --generator_models LIST       Comma-separated generator-model names to keep (overrides auto-infer).",
    "                                Example: \"gpt-4o,claude-3.5-sonnet\"",
    "  -h, --help                    Show this help",
    "",
    "Interpretation:",
    "  - p_item = mean correctness across LLMs for an item.",
    "  - Lower topic mean(p_item) => harder topic on average.",
    "  - Global permutation test asks whether topic labels explain variation in item difficulty.",
    "",
    sep = "\n"
  ))
}

parse_args <- function(args) {
  opts <- list()
  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      opts[["help"]] <- TRUE
      i <- i + 1
      next
    }
    if (grepl("^--", arg)) {
      if (grepl("=", arg)) {
        key <- sub("^--", "", sub("=.*", "", arg))
        val <- sub("^--[^=]+=", "", arg)
        opts[[key]] <- val
        i <- i + 1
        next
      }
      key <- sub("^--", "", arg)
      if (i == length(args) || grepl("^--", args[[i + 1]])) {
        opts[[key]] <- TRUE
        i <- i + 1
        next
      }
      opts[[key]] <- args[[i + 1]]
      i <- i + 2
      next
    }
    stop(paste("Unrecognized argument:", arg))
  }
  opts
}

as_int_or_stop <- function(x, name, default) {
  if (is.null(x) || x == "") return(as.integer(default))
  v <- suppressWarnings(as.integer(x))
  if (is.na(v) || v <= 0) stop(paste("Invalid", name, ":", x))
  v
}

# Robust boolean parser:
# Accepts: TRUE/FALSE, T/F, true/false, 1/0, yes/no, y/n
as_bool <- function(x, default = FALSE) {
  if (is.null(x) || x == "") return(default)
  if (is.logical(x)) return(isTRUE(x))
  s <- tolower(trimws(as.character(x)))
  if (s %in% c("true", "t", "1", "yes", "y")) return(TRUE)
  if (s %in% c("false", "f", "0", "no", "n")) return(FALSE)
  # If flag was provided without value, parse_args sets it to TRUE already.
  if (identical(x, TRUE)) return(TRUE)
  default
}

read_result_file <- function(path) {
  base <- basename(path)
  if (grepl("_judge_result\\.json$", base)) return(NULL)
  model <- sub("_result\\.json$", "", base)

  items <- tryCatch(jsonlite::fromJSON(path), error = function(e) NULL)
  if (is.null(items)) return(NULL)

  # fromJSON may return a data.frame or list; try to coerce to data.frame
  if (!is.data.frame(items)) {
    items <- tryCatch(as.data.frame(items), error = function(e) NULL)
  }
  if (is.null(items) || !is.data.frame(items) || nrow(items) == 0) return(NULL)

  # QuizBench convention: runs_root/<generator>/<quiz_id>/<model>_result.json
  quiz_id <- basename(dirname(path))
  quiz_form <- basename(dirname(dirname(path)))
  if (is.null(quiz_form) || quiz_form == "") quiz_form <- "unknown"

  qid  <- items[["question_id"]]
  gold <- items[["answer"]]
  pred <- items[["pred"]]

  topic <- items[["abms_specialty"]]
  if (is.null(topic)) topic <- items[["target_topic"]]
  if (is.null(topic)) topic <- items[["topic"]]

  if (is.null(qid) || is.null(gold)) return(NULL)

  pred_str <- ifelse(is.na(pred), "", as.character(pred))
  gold_str <- as.character(gold)
  y <- ifelse(pred_str == gold_str, 1L, 0L)

  # Make item IDs robust to cross-quiz collisions:
  item_id <- paste0(as.character(quiz_id), "::", as.character(qid))

  data.frame(
    item   = item_id,
    llm    = as.character(model),
    y      = as.integer(y),
    topic  = if (is.null(topic)) NA_character_ else as.character(topic),
    quiz   = as.character(quiz_form),
    quiz_id = as.character(quiz_id),
    stringsAsFactors = FALSE
  )
}

load_df_from_runs <- function(runs_root) {
  paths <- list.files(runs_root, pattern = "_result\\.json$", recursive = TRUE, full.names = TRUE)
  paths <- paths[!grepl("_judge_result\\.json$", paths)]
  if (length(paths) == 0) stop(paste("No *_result.json files found under:", runs_root))

  dfs <- lapply(paths, read_result_file)
  dfs <- dfs[!vapply(dfs, is.null, FUN.VALUE = logical(1))]
  if (length(dfs) == 0) stop("No usable result rows parsed from *_result.json files.")
  do.call(rbind, dfs)
}

# ---------- main ----------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0 || any(args %in% c("-h", "--help"))) {
  usage()
  quit(status = 0)
}
opts <- parse_args(args)
if (!is.null(opts[["help"]])) {
  usage()
  quit(status = 0)
}

timestamp <- format(Sys.time(), "%Y%m%dT%H%M%S")
outdir <- opts[["outdir"]]
if (is.null(outdir) || outdir == "") {
  outdir <- file.path("R_results", paste0("topic_difficulty_", timestamp))
}
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

runs_root <- opts[["runs_root"]]
input_csv <- opts[["input_csv"]]
if ((is.null(runs_root) || runs_root == "") && (is.null(input_csv) || input_csv == "")) {
  stop("Provide either --runs_root PATH or --input_csv PATH")
}
if (!is.null(input_csv) && input_csv != "" && !file.exists(input_csv)) {
  stop(paste("Input CSV not found:", input_csv))
}
if (!is.null(runs_root) && runs_root != "" && !dir.exists(runs_root)) {
  stop(paste("runs_root not found:", runs_root))
}

boot_reps <- as_int_or_stop(opts[["boot_reps"]], "--boot_reps", 5000)
perm_reps <- as_int_or_stop(opts[["perm_reps"]], "--perm_reps", 5000)
seed <- as_int_or_stop(opts[["seed"]], "--seed", 2026)

drop_incomplete <- as_bool(opts[["drop_incomplete_items"]], default = TRUE)
perm_unstratified <- as_bool(opts[["perm_unstratified"]], default = FALSE)
min_items_per_topic <- as_int_or_stop(opts[["min_items_per_topic"]], "--min_items_per_topic", 5)

generator_only <- as_bool(opts[["generator_only"]], default = FALSE)
generator_models_arg <- opts[["generator_models"]]

df <- NULL
if (!is.null(input_csv) && input_csv != "") {
  df <- read.csv(input_csv, stringsAsFactors = FALSE)
} else {
  df <- load_df_from_runs(runs_root)
  # Save parsed item-level file for reproducibility
  parsed_path <- file.path(outdir, "item_level_parsed.csv")
  write.csv(df, parsed_path, row.names = FALSE)
}

# Validate required columns
required_cols <- c("item", "llm", "y", "topic")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

# Clean
df <- df |>
  mutate(
    item = as.character(item),
    llm = as.character(llm),
    y = as.integer(y),
    topic = ifelse(is.na(topic) | topic == "", NA_character_, as.character(topic)),
    quiz = if ("quiz" %in% names(df)) ifelse(is.na(quiz) | quiz == "", NA_character_, as.character(quiz)) else NA_character_
  ) |>
  filter(!is.na(item) & item != "" & !is.na(llm) & llm != "" & !is.na(y))

if (nrow(df) == 0) stop("No rows left after cleaning.")


# ---- Optional: restrict to generator-model LLMs only ----
if (generator_only) {
  gens <- NULL

  # If user explicitly provided generator model names, use those.
  if (!is.null(generator_models_arg) && generator_models_arg != "") {
    gens <- trimws(unlist(strsplit(as.character(generator_models_arg), ",")))
    gens <- gens[gens != ""]
  } else {
    # Otherwise infer generator names from df$quiz (runs_root parsing provides this)
    if (!("quiz" %in% names(df)) || all(is.na(df$quiz))) {
      stop(
        "generator_only=TRUE but no generator labels are available.\n",
        "If using --input_csv, either include a 'quiz' column (generator name per row) ",
        "or pass --generator_models \"name1,name2,...\"."
      )
    }
    gens <- sort(unique(df$quiz[!is.na(df$quiz)]))
  }

  before_rows <- nrow(df)
  before_llms <- length(unique(df$llm))

  df <- df |> filter(llm %in% gens)

  after_rows <- nrow(df)
  after_llms <- length(unique(df$llm))

  # Save the set used for reproducibility
  writeLines(gens, con = file.path(outdir, "generator_models_used.txt"))

  cat(sprintf(
    "[INFO] generator_only=TRUE. Kept %d/%d rows; LLMs %d -> %d.\n",
    after_rows, before_rows, before_llms, after_llms
  ))

  if (nrow(df) == 0) {
    stop(
      "No rows remain after restricting to generator models.\n",
      "This usually means generator names (directory labels) do not match answer-model filenames.\n",
      "Fix by passing --generator_models with the exact llm names used in your *_result.json filenames."
    )
  }
}


# Ensure one response per (item,llm)
dup_counts <- df |>
  count(item, llm, name = "n") |>
  filter(n > 1)
if (nrow(dup_counts) > 0) {
  warning(paste0("Found duplicate (item,llm) rows; keeping first occurrence for each. Examples:\n",
                 paste(utils::capture.output(head(dup_counts, 10)), collapse = "\n")))
  df <- df |>
    group_by(item, llm) |>
    slice(1) |>
    ungroup()
}

# Pivot to wide to preserve paired-by-item structure across LLMs
wide <- df |>
  select(item, llm, y) |>
  pivot_wider(names_from = llm, values_from = y)

llms <- setdiff(names(wide), "item")
if (length(llms) < 1) stop("Need at least 1 LLM after pivoting.")

X <- as.matrix(wide[, llms, drop = FALSE])
storage.mode(X) <- "numeric"

complete_items <- complete.cases(X)
if (!all(complete_items)) {
  n_drop <- sum(!complete_items)
  msg <- paste0("Found ", n_drop, " item(s) with missing responses across LLMs.")
  if (drop_incomplete) {
    warning(paste0(msg, " Dropping them (recommended for a consistent p_item denominator)."))
    X <- X[complete_items, , drop = FALSE]
    wide <- wide[complete_items, , drop = FALSE]
  } else {
    warning(paste0(msg, " Keeping them and treating NA as incorrect (0)."))
    X[is.na(X)] <- 0
  }
}

n_items <- nrow(X)
if (n_items <= 1) stop("Not enough items after filtering.")

# Item metadata (topic/quiz per item)
item_meta <- df |>
  select(item, topic, quiz) |>
  group_by(item) |>
  summarise(
    topic = {
      u <- unique(topic[!is.na(topic)])
      if (length(u) == 0) NA_character_ else u[[1]]
    },
    quiz = {
      u <- unique(quiz[!is.na(quiz)])
      if (length(u) == 0) NA_character_ else u[[1]]
    },
    topic_n_unique = length(unique(topic[!is.na(topic)])),
    quiz_n_unique = length(unique(quiz[!is.na(quiz)])),
    .groups = "drop"
  ) |>
  filter(item %in% wide$item)

if (any(item_meta$topic_n_unique > 1)) {
  warning("Some items had inconsistent topic labels across rows; using the first non-missing topic per item.")
}
if (any(item_meta$quiz_n_unique > 1)) {
  warning("Some items had inconsistent quiz labels across rows; using the first non-missing quiz per item.")
}

# Compute item difficulty score: p_item = mean correctness across LLMs for that item
item_scores <- data.frame(
  item = wide$item,
  p_item = rowMeans(X),
  n_llms = ncol(X),
  stringsAsFactors = FALSE
) |>
  left_join(item_meta |> select(item, topic, quiz), by = "item")

item_scores_path <- file.path(outdir, "item_scores.csv")
write.csv(item_scores, item_scores_path, row.names = FALSE)

# Keep only items with topic labels
item_scores_topic <- item_scores |>
  filter(!is.na(topic) & topic != "")

if (nrow(item_scores_topic) == 0) {
  stop("No items with non-missing topic labels; cannot analyze topic difficulty.")
}

# Filter out very small topics (pre-specified rule)
topic_counts <- item_scores_topic |>
  count(topic, name = "n_items")

small_topics <- topic_counts |>
  filter(n_items < min_items_per_topic)

if (nrow(small_topics) > 0) {
  warning(paste0(
    "Dropping topics with < ", min_items_per_topic, " items: ",
    paste(small_topics$topic, collapse = ", ")
  ))
}

item_scores_topic <- item_scores_topic |>
  left_join(topic_counts, by = "topic") |>
  filter(n_items >= min_items_per_topic)

if (length(unique(item_scores_topic$topic)) < 2) {
  stop("Need at least 2 topics (after filtering) to test whether topics differ in difficulty.")
}

# ---------- Topic means + bootstrap CIs ----------
set.seed(seed)

topics <- sort(unique(item_scores_topic$topic))
topic_table <- item_scores_topic |>
  group_by(topic) |>
  summarise(
    n_items = n(),
    mean_p_item = mean(p_item),
    .groups = "drop"
  ) |>
  arrange(mean_p_item)  # lower = harder

# Bootstrap CI per topic (resample items within topic)
boot_lo <- numeric(length(topics))
boot_hi <- numeric(length(topics))

for (i in seq_along(topics)) {
  t <- topics[[i]]
  v <- item_scores_topic$p_item[item_scores_topic$topic == t]
  n <- length(v)
  # Bootstrap mean within topic
  boot_means <- replicate(boot_reps, mean(sample(v, n, replace = TRUE)))
  qs <- as.numeric(quantile(boot_means, c(0.025, 0.975), names = FALSE))
  boot_lo[[i]] <- qs[[1]]
  boot_hi[[i]] <- qs[[2]]
}

topic_table <- topic_table |>
  mutate(
    lo95 = boot_lo[match(topic, topics)],
    hi95 = boot_hi[match(topic, topics)],
    mean_pct = 100 * mean_p_item,
    lo95_pct = 100 * lo95,
    hi95_pct = 100 * hi95
  ) |>
  arrange(mean_p_item) |>
  mutate(rank_hard_to_easy = row_number())

topic_table_path <- file.path(outdir, "topic_difficulty_bootstrap.csv")
write.csv(topic_table, topic_table_path, row.names = FALSE)

# ---------- Effect size: range between easiest and hardest topic ----------
# Use topic-stratified bootstrap (preserves per-topic n_items)
split_idx <- split(seq_len(nrow(item_scores_topic)), item_scores_topic$topic)
n_t <- vapply(split_idx, length, integer(1))

topic_means_obs <- vapply(split_idx, function(idx) mean(item_scores_topic$p_item[idx]), numeric(1))
range_obs <- max(topic_means_obs) - min(topic_means_obs)

boot_range <- numeric(boot_reps)
for (b in seq_len(boot_reps)) {
  means_b <- vapply(seq_along(split_idx), function(j) {
    idx <- split_idx[[j]]
    samp <- sample(idx, length(idx), replace = TRUE)
    mean(item_scores_topic$p_item[samp])
  }, numeric(1))
  boot_range[[b]] <- max(means_b) - min(means_b)
}
range_ci <- as.numeric(quantile(boot_range, c(0.025, 0.975), names = FALSE))

# ---------- Global permutation test (items are unit) ----------
anova_F <- function(values, groups) {
  groups <- as.factor(groups)
  k <- nlevels(groups)
  n <- length(values)
  if (k < 2 || n <= k) return(list(F = NA_real_, eta_sq = NA_real_))
  grand <- mean(values)
  means <- tapply(values, groups, mean)
  ns <- tapply(values, groups, length)
  means_by_group <- means[as.character(groups)]
  ss_between <- sum(ns * (means - grand)^2)
  ss_within <- sum((values - means_by_group)^2)
  df1 <- k - 1
  df2 <- n - k
  Fstat <- (ss_between / df1) / (ss_within / df2)
  eta_sq <- ss_between / (ss_between + ss_within)
  list(F = as.numeric(Fstat), eta_sq = as.numeric(eta_sq))
}

permute_labels <- function(labels, strata = NULL) {
  if (is.null(strata)) return(sample(labels))
  strata <- ifelse(is.na(strata) | strata == "", "unknown", as.character(strata))
  out <- labels
  for (s in unique(strata)) {
    idx <- which(strata == s)
    out[idx] <- sample(labels[idx])
  }
  out
}

values <- item_scores_topic$p_item
groups <- item_scores_topic$topic

# Optional stratification by quiz (if available and useful)
strata <- NULL
stratified <- FALSE
if (!perm_unstratified && "quiz" %in% names(item_scores_topic)) {
  q <- item_scores_topic$quiz
  if (!all(is.na(q)) && length(unique(q[!is.na(q)])) > 1) {
    strata <- q
    stratified <- TRUE
  }
}

obs <- anova_F(values, groups)
F_obs <- obs$F
eta_obs <- obs$eta_sq

F_perm <- numeric(perm_reps)
for (b in seq_len(perm_reps)) {
  g_perm <- permute_labels(groups, strata = strata)
  F_perm[[b]] <- anova_F(values, g_perm)$F
}
p_perm <- (1 + sum(F_perm >= F_obs, na.rm = TRUE)) / (perm_reps + 1)

global <- data.frame(
  n_items = nrow(item_scores_topic),
  n_topics = length(unique(groups)),
  n_llms = ncol(X),
  stratified_by_quiz = stratified,
  boot_reps = boot_reps,
  perm_reps = perm_reps,
  F_stat = F_obs,
  eta_squared = eta_obs,
  p_value_perm = p_perm,
  range_topic_means_pp = 100 * range_obs,
  range_lo95_pp = 100 * range_ci[1],
  range_hi95_pp = 100 * range_ci[2],
  stringsAsFactors = FALSE
)

global_path <- file.path(outdir, "topic_global_test_permutation.csv")
write.csv(global, global_path, row.names = FALSE)

# ---------- Summary ----------
hardest <- topic_table[1, ]
easiest <- topic_table[nrow(topic_table), ]

summary_lines <- c(
  "Topic difficulty only (item is the unit; p_item = mean correctness across LLMs).",
  "",
  paste0("Items used (with topic): ", nrow(item_scores_topic)),
  paste0("Topics included: ", length(unique(item_scores_topic$topic))),
  paste0("LLMs contributing to p_item: ", ncol(X)),
  paste0("Dropped incomplete items: ", drop_incomplete),
  paste0("Permutation stratified by quiz: ", stratified),
  "",
  "Hardest topic (lowest mean p_item):",
  paste0("  ", hardest$topic, ": ", sprintf("%.1f", hardest$mean_pct), "% (95% CI ",
         sprintf("%.1f", hardest$lo95_pct), "–", sprintf("%.1f", hardest$hi95_pct), "%), n=", hardest$n_items),
  "",
  "Easiest topic (highest mean p_item):",
  paste0("  ", easiest$topic, ": ", sprintf("%.1f", easiest$mean_pct), "% (95% CI ",
         sprintf("%.1f", easiest$lo95_pct), "–", sprintf("%.1f", easiest$hi95_pct), "%), n=", easiest$n_items),
  "",
  "Global test (permutation ANOVA on item scores by topic):",
  paste0("  F = ", signif(F_obs, 4), ", eta^2 = ", signif(eta_obs, 4),
         ", p_perm = ", signif(p_perm, 4)),
  "",
  "Effect size (range of topic mean item scores):",
  paste0("  Range = ", sprintf("%.1f", 100 * range_obs), " pp (95% bootstrap CI ",
         sprintf("%.1f", 100 * range_ci[1]), "–", sprintf("%.1f", 100 * range_ci[2]), " pp)"),
  "",
  "Files written:",
  paste0("  - ", normalizePath(item_scores_path, winslash = "/")),
  paste0("  - ", normalizePath(topic_table_path, winslash = "/")),
  paste0("  - ", normalizePath(global_path, winslash = "/"))
)

writeLines(summary_lines, con = file.path(outdir, "topic_difficulty_summary.txt"))

cat(paste0("[OK] Wrote:\n  ", item_scores_path, "\n  ", topic_table_path, "\n  ", global_path, "\n  ",
           file.path(outdir, "topic_difficulty_summary.txt"), "\n"))
