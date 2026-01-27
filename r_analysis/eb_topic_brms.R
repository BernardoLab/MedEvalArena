#!/usr/bin/env Rscript
# Empirical Bayes (partial pooling) content-area subscores via brms.
#
# This script fits a Bayesian multilevel logistic model to item-level responses:
#   y ~ 1 + topic + (1 + topic || llm) + (1|quiz) + (1|item)
#
# Where:
#   - y: 0/1 correctness per (LLM, item)
#   - llm: answer model (e.g., gpt-*, claude-*, gemini-*)
#   - topic: content area (e.g., ABMS specialty)
#   - quiz: form / generator (e.g., 6 generator models for ABMS subsets)
#   - item: unique question identifier (accounts for item difficulty)
#
# It then extracts “EB content area estimates” as predicted % correct per (llm, topic),
# averaging out item and quiz idiosyncrasies by excluding those random effects during
# prediction (Option A: tidybayes::add_epred_draws()).
#
# Example (ABMS subset runs):
#   Rscript r_analysis/eb_topic_brms.R \
#     --runs_root eval_results/quizbench/quizzes_ABMS20260101/runs
#
# Outputs are written under R_results/eb_topic_brms_<timestamp>/ by default.

required_pkgs <- c("brms", "tidybayes", "dplyr", "tidyr", "ggplot2", "jsonlite")
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
  library(brms)
  library(tidybayes)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(jsonlite)
})

usage <- function() {
  cat(paste(
    "Empirical Bayes (partial pooling) topic subscores via brms.",
    "",
    "Input options (choose one):",
    "  --runs_root PATH         QuizBench runs root containing *_result.json files",
    "  --input_csv PATH         Pre-built CSV with columns: y, llm, topic, quiz, item",
    "",
    "Optional:",
    "  --outdir PATH            Output directory (default: R_results/eb_topic_brms_<timestamp>)",
    "  --fit_rds PATH           Where to save/load the fitted brms model (default: <outdir>/brms_fit.rds)",
    "  --reuse_fit              If set, reuse --fit_rds if it exists (skip refit)",
    "  --chains N               brms chains (default: 4)",
    "  --cores N                brms cores (default: 4)",
    "  --iter N                 brms iterations per chain (default: 4000)",
    "  --seed N                 RNG seed (default: 2026)",
    "  -h, --help               Show this help",
    "",
    "Notes:",
    "  - For ABMS subset runs, topic is read from abms_specialty/target_topic in *_result.json.",
    "  - For non-ABMS runs, ensure your input CSV includes topic labels.",
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
  outdir <- file.path("R_results", paste0("eb_topic_brms_", timestamp))
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

chains <- as_int_or_stop(opts[["chains"]], "--chains", 4)
cores <- as_int_or_stop(opts[["cores"]], "--cores", 4)
iter <- as_int_or_stop(opts[["iter"]], "--iter", 4000)
seed <- as_int_or_stop(opts[["seed"]], "--seed", 2026)

reuse_fit <- isTRUE(opts[["reuse_fit"]])
fit_rds <- opts[["fit_rds"]]
if (is.null(fit_rds) || fit_rds == "") {
  fit_rds <- file.path(outdir, "brms_fit.rds")
}

read_result_file <- function(path) {
  model <- sub("_result\\.json$", "", basename(path))
  if (grepl("_judge_result\\.json$", basename(path))) return(NULL)

  items <- tryCatch(jsonlite::fromJSON(path), error = function(e) NULL)
  if (is.null(items) || !is.data.frame(items) || nrow(items) == 0) return(NULL)

  # Expect runs_root/<quiz>/<quiz_id>/<model>_result.json (QuizBench convention).
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
    y = as.integer(y),
    llm = as.character(model),
    topic = if (is.null(topic)) NA_character_ else as.character(topic),
    quiz = as.character(quiz_form),
    item = as.character(qid),
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

required_cols <- c("y", "llm", "topic", "quiz", "item")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

df <- df |>
  transform(
    y = as.integer(y),
    llm = factor(llm),
    topic = factor(topic),
    quiz = factor(quiz),
    item = factor(item)
  )

df <- df[!is.na(df$y) & !is.na(df$llm) & !is.na(df$topic) & !is.na(df$quiz) & !is.na(df$item), ]
if (nrow(df) == 0) stop("No rows left after filtering missing values in required columns.")

data_summary <- c(
  paste("rows:", nrow(df)),
  paste("llm_levels:", nlevels(df$llm)),
  paste("topic_levels:", nlevels(df$topic)),
  paste("quiz_levels:", nlevels(df$quiz)),
  paste("item_levels:", nlevels(df$item))
)
writeLines(data_summary, con = file.path(outdir, "data_summary.txt"))

fit <- NULL
if (reuse_fit && file.exists(fit_rds)) {
  message(paste("[INFO] Loading existing fit from", fit_rds))
  fit <- readRDS(fit_rds)
} else {
  message("[INFO] Fitting brms model (this can take a while)...")

  f <- bf(y ~ 1 + topic + (1 + topic || llm) + (1 | quiz) + (1 | item))
  pri <- c(
    prior(normal(0, 1.5), class = Intercept),
    prior(normal(0, 1.0), class = b),
    prior(exponential(1), class = sd)
  )

  fit <- brm(
    formula = f,
    data = df,
    family = bernoulli(link = "logit"),
    prior = pri,
    chains = chains,
    cores = cores,
    iter = iter,
    seed = seed
  )
  saveRDS(fit, fit_rds)
}

capture.output(summary(fit), file = file.path(outdir, "brms_summary.txt"))

grid <- tidyr::expand_grid(
  llm = levels(df$llm),
  topic = levels(df$topic),
  quiz = levels(df$quiz)[1],
  item = levels(df$item)[1]
)
grid$llm <- factor(grid$llm, levels = levels(df$llm))
grid$topic <- factor(grid$topic, levels = levels(df$topic))
grid$quiz <- factor(grid$quiz, levels = levels(df$quiz))
grid$item <- factor(grid$item, levels = levels(df$item))

eb_topic <- fit |>
  add_epred_draws(
    newdata = grid,
    re_formula = ~(1 + topic || llm)
  ) |>
  group_by(llm, topic) |>
  median_qi(.epred, .width = 0.95) |>
  ungroup() |>
  transmute(
    llm = as.character(llm),
    topic = as.character(topic),
    eb_pct = 100 * .epred,
    lo95 = 100 * .lower,
    hi95 = 100 * .upper
  )

raw_topic <- df |>
  group_by(llm, topic) |>
  summarise(
    n = n(),
    raw_pct = 100 * mean(y),
    .groups = "drop"
  ) |>
  mutate(
    llm = as.character(llm),
    topic = as.character(topic)
  )

compare <- raw_topic |>
  left_join(eb_topic, by = c("llm", "topic")) |>
  mutate(shrinkage_points = eb_pct - raw_pct) |>
  arrange(topic, llm)

write.csv(eb_topic, file.path(outdir, "eb_topic.csv"), row.names = FALSE)
write.csv(compare, file.path(outdir, "raw_vs_eb_topic.csv"), row.names = FALSE)

plot_path <- file.path(outdir, "eb_topic_pointrange.png")
p <- ggplot(eb_topic, aes(x = topic, y = eb_pct, ymin = lo95, ymax = hi95)) +
  geom_pointrange() +
  facet_wrap(~ llm) +
  coord_cartesian(ylim = c(0, 100)) +
  labs(
    x = "Content area (topic)",
    y = "Estimated % correct (Empirical Bayes / partial pooling)",
    caption = "Points = posterior median; bars = 95% credible interval. Item and quiz effects set to 0."
  ) +
  theme_bw(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.minor = element_blank()
  )
ggsave(plot_path, p, width = 14, height = 10, dpi = 200)

message(paste("[OK] Wrote outputs to", outdir))
message(paste("  -", file.path(outdir, "eb_topic.csv")))
message(paste("  -", file.path(outdir, "raw_vs_eb_topic.csv")))
message(paste("  -", plot_path))
message(paste("  - fit:", fit_rds))

