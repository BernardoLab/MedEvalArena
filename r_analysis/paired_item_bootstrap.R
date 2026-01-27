#!/usr/bin/env Rscript
# Paired item-block bootstrap for LLM accuracy comparisons.
#
# This script implements a simple, nonparametric analysis for QuizBench-style
# experiments where all LLMs answer the same set of items:
#   1) Overall accuracy per LLM + 95% CI via bootstrap over items
#   2) Pairwise paired differences (A - B) + 95% CI via the same bootstrap draws
#   3) A pre-specified "major difference" threshold Delta (percentage points),
#      and a flag whether each pairwise CI lies within ±Delta.
#
# For ABMS subset runs (e.g., ABMS20260101), topic labels are sourced from the
# per-item fields embedded in *_result.json (abms_specialty / target_topic).
#
# Example:
#   Rscript r_analysis/paired_item_bootstrap.R \
#     --runs_root eval_results/quizbench/quizzes_ABMS20260101/runs \
#     --outdir R_results/paired_bootstrap_ABMS20260101 \
#     --boot_reps 5000 \
#     --delta 5 \
#     --seed 2026

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
    "Paired item-block bootstrap accuracy comparisons for QuizBench runs.",
    "",
    "Input options (choose one):",
    "  --runs_root PATH         QuizBench runs root containing *_result.json files",
    "  --input_csv PATH         Pre-built CSV with columns: item, llm, y, topic (topic optional)",
    "",
    "Optional:",
    "  --outdir PATH            Output directory (default: R_results/paired_item_bootstrap_<timestamp>)",
    "  --boot_reps N            Bootstrap replicates (default: 5000)",
    "  --delta D                Major-difference threshold in percentage points (default: 5)",
    "  --seed N                 RNG seed (default: 2026)",
    "  --drop_incomplete_items  Drop items with any missing LLM response (default: true)",
    "  --include_all_llms       Include all answer models (default: only quiz-generator models when possible)",
    "",
    "Topic pooling / global test (optional, runs if topic exists):",
    "  --topic_perm_reps N         Permutation reps for global topic test (default: 5000)",
    "  --topic_perm_unstratified   Permute topics across all items (default: stratify within quiz when available)",
    "  --skip_topic_global_test    Skip global topic test + pooled topic table",
    "  -h, --help               Show this help",
    "",
    "Notes:",
    "  - Bootstrapping resamples items with replacement, preserving paired structure across LLMs.",
    "  - For ABMS subset runs, topic is read from abms_specialty/target_topic in *_result.json.",
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
  outdir <- file.path("R_results", paste0("paired_item_bootstrap_", timestamp))
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

as_int_or_stop <- function(x, name, default) {
  if (is.null(x) || x == "") return(as.integer(default))
  v <- suppressWarnings(as.integer(x))
  if (is.na(v) || v <= 0) stop(paste("Invalid", name, ":", x))
  v
}

as_num_or_stop <- function(x, name, default) {
  if (is.null(x) || x == "") return(as.numeric(default))
  v <- suppressWarnings(as.numeric(x))
  if (is.na(v) || !is.finite(v)) stop(paste("Invalid", name, ":", x))
  v
}

boot_reps <- as_int_or_stop(opts[["boot_reps"]], "--boot_reps", 5000)
if (!is.null(opts[["B_reps"]])) {
  boot_reps <- as_int_or_stop(opts[["B_reps"]], "--B_reps", boot_reps)
}
delta <- as_num_or_stop(opts[["delta"]], "--delta", 5)
if (delta <= 0) stop("--delta must be > 0")
seed <- as_int_or_stop(opts[["seed"]], "--seed", 2026)
drop_incomplete <- TRUE
if (!is.null(opts[["drop_incomplete_items"]])) {
  drop_incomplete <- isTRUE(opts[["drop_incomplete_items"]])
}
include_all_llms <- isTRUE(opts[["include_all_llms"]])
topic_perm_reps <- as_int_or_stop(opts[["topic_perm_reps"]], "--topic_perm_reps", 5000)
topic_perm_unstratified <- isTRUE(opts[["topic_perm_unstratified"]])
skip_topic_global_test <- isTRUE(opts[["skip_topic_global_test"]])

read_result_file <- function(path) {
  model <- sub("_result\\.json$", "", basename(path))
  if (grepl("_judge_result\\.json$", basename(path))) return(NULL)

  items <- tryCatch(jsonlite::fromJSON(path), error = function(e) NULL)
  if (is.null(items) || !is.data.frame(items) || nrow(items) == 0) return(NULL)

  # Expect runs_root/<generator>/<quiz_id>/<model>_result.json (QuizBench convention).
  quiz_id <- basename(dirname(path))
  quiz_form <- basename(dirname(dirname(path)))
  if (is.null(quiz_form) || quiz_form == "") quiz_form <- "unknown"

  qid <- items[["question_id"]]
  gold <- items[["answer"]]
  pred <- items[["pred"]]
  topic <- items[["abms_specialty"]]
  if (is.null(topic)) topic <- items[["target_topic"]]
  if (is.null(topic)) topic <- items[["topic"]]

  if (is.null(qid) || is.null(gold)) return(NULL)

  # Treat missing/unparseable pred as incorrect (0), matching eval_quiz.py behavior.
  pred_str <- ifelse(is.na(pred), "", as.character(pred))
  gold_str <- as.character(gold)
  y <- ifelse(pred_str == gold_str, 1L, 0L)

  out <- data.frame(
    item = as.character(qid),
    llm = as.character(model),
    y = as.integer(y),
    topic = if (is.null(topic)) NA_character_ else as.character(topic),
    quiz = as.character(quiz_form),
    quiz_id = as.character(quiz_id),
    stringsAsFactors = FALSE
  )
  out
}

load_df_from_runs <- function(runs_root) {
  paths <- list.files(runs_root, pattern = "_result\\.json$", recursive = TRUE, full.names = TRUE)
  paths <- paths[!grepl("_judge_result\\.json$", paths)]
  if (length(paths) == 0) stop(paste("No *_result.json files found under:", runs_root))

  dfs <- lapply(paths, read_result_file)
  dfs <- dfs[!vapply(dfs, is.null, FUN.VALUE = logical(1))]
  if (length(dfs) == 0) stop("No usable result rows parsed from *_result.json files.")

  df <- do.call(rbind, dfs)
  df
}

df <- NULL
if (!is.null(input_csv) && input_csv != "") {
  df <- read.csv(input_csv, stringsAsFactors = FALSE)
} else {
  df <- load_df_from_runs(runs_root)
  input_csv <- file.path(outdir, "item_level_with_topics.csv")
  write.csv(df, input_csv, row.names = FALSE)
}

required_cols <- c("item", "llm", "y")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

df <- df |>
  transform(
    item = as.character(item),
    llm = as.character(llm),
    y = as.integer(y)
  )

df <- df[!is.na(df$item) & df$item != "" & !is.na(df$llm) & df$llm != "" & !is.na(df$y), ]
if (nrow(df) == 0) stop("No rows left after filtering missing values in required columns.")

if (!include_all_llms) {
  if ("quiz" %in% names(df)) {
    generator_models <- unique(df$quiz[!is.na(df$quiz) & df$quiz != ""])
    if (length(generator_models) > 0) {
      df <- df |>
        filter(llm %in% generator_models)
      if (nrow(df) == 0) {
        stop(paste0(
          "After restricting to quiz-generator LLMs, no rows remain.\n",
          "If your dataset does not include generator models as answer models, re-run with --include_all_llms."
        ))
      }
      cat(paste0("[INFO] LLM scope: quiz-generator models only (", length(unique(df$llm)), " model(s)).\n"))
    } else {
      warning("Could not infer generator models from df$quiz; keeping all LLMs. (Use --include_all_llms to silence this.)")
    }
  } else {
    warning("No 'quiz' column available to infer generator models; keeping all LLMs. (Use --include_all_llms to silence this.)")
  }
} else {
  cat(paste0("[INFO] LLM scope: all answer models (", length(unique(df$llm)), " model(s)).\n"))
}

# Ensure one response per (item, llm).
dup_counts <- df |>
  as.data.frame() |>
  count(item, llm, name = "n") |>
  filter(n > 1)
if (nrow(dup_counts) > 0) {
  warning(paste0("Found duplicate (item,llm) rows; keeping first occurrence for each.\n",
                 "Examples:\n",
                 paste(utils::capture.output(head(dup_counts, 10)), collapse = "\n")))
  df <- df |>
    group_by(item, llm) |>
    slice(1) |>
    ungroup()
}

# Wide matrix: one row per item, one column per LLM (paired).
wide <- df |>
  select(item, llm, y) |>
  pivot_wider(names_from = llm, values_from = y)

llms <- setdiff(names(wide), "item")
if (length(llms) < 2) stop("Need at least 2 LLM columns after pivoting to compute pairwise differences.")

X <- as.matrix(wide[, llms, drop = FALSE])
storage.mode(X) <- "numeric"

complete_items <- complete.cases(X)
if (!all(complete_items)) {
  n_drop <- sum(!complete_items)
  msg <- paste0("Found ", n_drop, " item(s) with missing responses across LLMs.")
  if (drop_incomplete) {
    warning(paste0(msg, " Dropping them (paired bootstrap requires paired items)."))
    X <- X[complete_items, , drop = FALSE]
    wide <- wide[complete_items, , drop = FALSE]
  } else {
    warning(paste0(msg, " Keeping them and treating NA as incorrect (0)."))
    X[is.na(X)] <- 0
  }
}

n_items <- nrow(X)
if (n_items <= 1) stop("Not enough paired items after filtering to run bootstrap.")

# Canonical long dataset aligned to X/wide, so all downstream tables/tests use
# the same item set and NA handling.
item_meta <- df |>
  select(item, topic, quiz) |>
  mutate(
    topic = ifelse(is.na(topic) | topic == "", NA_character_, as.character(topic)),
    quiz = ifelse(is.na(quiz) | quiz == "", NA_character_, as.character(quiz))
  ) |>
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
  )
if (any(item_meta$topic_n_unique > 1)) {
  ex <- item_meta[item_meta$topic_n_unique > 1, c("item", "topic_n_unique")]
  warning(paste0(
    "Found items with inconsistent topic labels across rows; using first non-missing topic per item.\n",
    paste(utils::capture.output(head(ex, 10)), collapse = "\n")
  ))
}
if (any(item_meta$quiz_n_unique > 1)) {
  ex <- item_meta[item_meta$quiz_n_unique > 1, c("item", "quiz_n_unique")]
  warning(paste0(
    "Found items with inconsistent quiz labels across rows; using first non-missing quiz per item.\n",
    paste(utils::capture.output(head(ex, 10)), collapse = "\n")
  ))
}
item_meta <- item_meta |>
  select(item, topic, quiz) |>
  filter(item %in% wide$item)

df_long <- data.frame(
  item = rep(wide$item, times = length(llms)),
  llm = rep(llms, each = n_items),
  y = as.integer(as.vector(X)),
  stringsAsFactors = FALSE
)
df_long <- df_long |>
  left_join(item_meta, by = "item")

set.seed(seed)
m <- ncol(X)
boot_acc <- matrix(NA_real_, nrow = boot_reps, ncol = m)
colnames(boot_acc) <- llms

for (b in seq_len(boot_reps)) {
  idx <- sample.int(n_items, n_items, replace = TRUE)
  boot_acc[b, ] <- colMeans(X[idx, , drop = FALSE])
}

point_acc <- colMeans(X)
ci_lo <- apply(boot_acc, 2, quantile, probs = 0.025, names = FALSE)
ci_hi <- apply(boot_acc, 2, quantile, probs = 0.975, names = FALSE)

overall <- data.frame(
  llm = llms,
  acc = as.numeric(point_acc),
  acc_pct = 100 * as.numeric(point_acc),
  lo95 = 100 * as.numeric(ci_lo),
  hi95 = 100 * as.numeric(ci_hi),
  stringsAsFactors = FALSE
)
overall <- overall[order(-overall$acc_pct, overall$llm), ]

overall_path <- file.path(outdir, "overall_accuracy_bootstrap.csv")
write.csv(overall, overall_path, row.names = FALSE)

# Pairwise paired bootstrap differences (A - B).
pairs <- combn(llms, 2, simplify = FALSE)
pairwise_list <- lapply(pairs, function(p) {
  A <- p[[1]]
  B <- p[[2]]
  diffs <- boot_acc[, A] - boot_acc[, B]
  med <- median(diffs)
  lo <- as.numeric(quantile(diffs, 0.025, names = FALSE))
  hi <- as.numeric(quantile(diffs, 0.975, names = FALSE))
  data.frame(
    A = A,
    B = B,
    diff_pct = 100 * med,
    lo95 = 100 * lo,
    hi95 = 100 * hi,
    stringsAsFactors = FALSE
  )
})
pairwise <- do.call(rbind, pairwise_list)
pairwise$within_major <- (pairwise$lo95 > -delta) & (pairwise$hi95 < delta)
pairwise$abs_diff_pct <- abs(pairwise$diff_pct)
pairwise <- pairwise[order(-pairwise$abs_diff_pct, pairwise$A, pairwise$B), ]

pairwise_path <- file.path(outdir, "pairwise_accuracy_diffs_bootstrap.csv")
write.csv(pairwise, pairwise_path, row.names = FALSE)

# Topic-level: raw topic accuracies per LLM with Wilson 95% CIs (no formal testing).
wilson_ci <- function(k, n, alpha = 0.05) {
  if (n <= 0) return(c(NA_real_, NA_real_))
  z <- qnorm(1 - alpha / 2)
  phat <- k / n
  denom <- 1 + (z^2) / n
  center <- (phat + (z^2) / (2 * n)) / denom
  half <- (z * sqrt((phat * (1 - phat) / n) + (z^2) / (4 * n^2))) / denom
  lo <- max(0, center - half)
  hi <- min(1, center + half)
  c(lo, hi)
}

topic_path <- NULL
df_topic_long <- df_long |>
  filter(!is.na(topic) & topic != "")
if (nrow(df_topic_long) > 0) {
  topic_acc <- df_topic_long |>
    group_by(llm, topic) |>
    summarise(
      n = n(),
      k = sum(y),
      acc = k / n,
      .groups = "drop"
    )

  ci_mat <- mapply(wilson_ci, topic_acc$k, topic_acc$n)
  topic_acc$lo95 <- as.numeric(ci_mat[1, ])
  topic_acc$hi95 <- as.numeric(ci_mat[2, ])
  topic_acc <- topic_acc |>
    mutate(
      acc_pct = 100 * acc,
      lo95_pct = 100 * lo95,
      hi95_pct = 100 * hi95
    ) |>
    arrange(topic, desc(acc_pct), llm)

  topic_path <- file.path(outdir, "topic_accuracy_wilson.csv")
  write.csv(topic_acc, topic_path, row.names = FALSE)
}

topic_pooled_path <- NULL
topic_global_path <- NULL
if (!skip_topic_global_test && nrow(df_topic_long) > 0) {
  # Pooled topic accuracies (across included LLMs).
  topic_pooled <- df_topic_long |>
    group_by(topic) |>
    summarise(
      n_responses = n(),
      k = sum(y),
      acc = k / n_responses,
      n_items = n_distinct(item),
      .groups = "drop"
    )
  ci_mat <- mapply(wilson_ci, topic_pooled$k, topic_pooled$n_responses)
  topic_pooled$lo95 <- as.numeric(ci_mat[1, ])
  topic_pooled$hi95 <- as.numeric(ci_mat[2, ])
  topic_pooled <- topic_pooled |>
    mutate(
      acc_pct = 100 * acc,
      lo95_pct = 100 * lo95,
      hi95_pct = 100 * hi95
    ) |>
    arrange(acc_pct)

  topic_pooled_path <- file.path(outdir, "topic_accuracy_pooled_wilson.csv")
  write.csv(topic_pooled, topic_pooled_path, row.names = FALSE)

  # Global test: do topics differ when pooling LLMs? Use item-level means and a
  # permutation test (items are the resampling unit).
  item_scores <- data.frame(
    item = wide$item,
    p_item = rowMeans(X),
    stringsAsFactors = FALSE
  ) |>
    left_join(item_meta, by = "item") |>
    filter(!is.na(topic) & topic != "")

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

  strata <- NULL
  stratified <- FALSE
  if (!topic_perm_unstratified && "quiz" %in% names(item_scores)) {
    q <- item_scores$quiz
    if (!all(is.na(q)) && length(unique(q[!is.na(q)])) > 1) {
      strata <- q
      stratified <- TRUE
    }
  }

  obs <- anova_F(item_scores$p_item, item_scores$topic)
  F_obs <- obs$F
  eta_obs <- obs$eta_sq

  F_perm <- numeric(topic_perm_reps)
  for (b in seq_len(topic_perm_reps)) {
    g_perm <- permute_labels(item_scores$topic, strata = strata)
    F_perm[[b]] <- anova_F(item_scores$p_item, g_perm)$F
  }
  p_perm <- (1 + sum(F_perm >= F_obs, na.rm = TRUE)) / (topic_perm_reps + 1)

  topic_global <- data.frame(
    n_items = nrow(item_scores),
    n_topics = length(unique(item_scores$topic)),
    n_llms = length(llms),
    stratified_by_quiz = stratified,
    F_stat = F_obs,
    eta_squared = eta_obs,
    perm_reps = topic_perm_reps,
    p_value_perm = p_perm,
    stringsAsFactors = FALSE
  )
  topic_global_path <- file.path(outdir, "topic_global_test_permutation.csv")
  write.csv(topic_global, topic_global_path, row.names = FALSE)
}

summary_lines <- c(
  paste0("Paired item bootstrap (resample items)"),
  paste0("Items used: ", n_items),
  paste0("LLMs: ", length(llms)),
  paste0("Bootstrap reps: ", boot_reps),
  paste0("Delta (major difference threshold, pp): ", delta),
  paste0("All pairwise CIs within ±Delta: ", all(pairwise$within_major))
)
if (!all(pairwise$within_major)) {
  offenders <- pairwise[!pairwise$within_major, c("A", "B", "diff_pct", "lo95", "hi95")]
  summary_lines <- c(summary_lines, "", "Pairs with CI extending beyond ±Delta:", "")
  summary_lines <- c(summary_lines, utils::capture.output(print(head(offenders, 50), row.names = FALSE)))
}

summary_lines <- c(
  summary_lines,
  "",
  "Files written:",
  paste0("  - ", normalizePath(overall_path, winslash = "/")),
  paste0("  - ", normalizePath(pairwise_path, winslash = "/")),
  if (!is.null(topic_path)) paste0("  - ", normalizePath(topic_path, winslash = "/")) else NULL,
  if (!is.null(topic_pooled_path)) paste0("  - ", normalizePath(topic_pooled_path, winslash = "/")) else NULL,
  if (!is.null(topic_global_path)) paste0("  - ", normalizePath(topic_global_path, winslash = "/")) else NULL
)
writeLines(summary_lines, con = file.path(outdir, "bootstrap_summary.txt"))

cat(paste0("[OK] Wrote:\n  ", overall_path, "\n  ", pairwise_path, "\n"))
if (!is.null(topic_path)) {
  cat(paste0("  ", topic_path, "\n"))
}
if (!is.null(topic_pooled_path)) {
  cat(paste0("  ", topic_pooled_path, "\n"))
}
if (!is.null(topic_global_path)) {
  cat(paste0("  ", topic_global_path, "\n"))
}
cat(paste0("  ", file.path(outdir, "bootstrap_summary.txt"), "\n"))
