#!/usr/bin/env Rscript
# Assess whether validity rates differ by generator.
#
# This uses the same per-generator counts produced in quizbench/aggregate_results.py:
#   - n_total: generator_total_counts[generator] (sum of n_items across runs)
#   - n_valid: generator_valid_counts[generator] (sum of questions passing judge filters)
#
# Note: The "_EXAMPLE" generator is excluded from the analysis by default.
#
# Usage:
#   Rscript r_analysis/validity_by_generator.R --counts_csv path/to/counts.csv
#   Rscript r_analysis/validity_by_generator.R --agg_log path/to/aggregate_results_stdout.txt
#   Rscript r_analysis/validity_by_generator.R --agg_log path/to/aggregate_results_stdout.txt --quiz_batch_tag Jan2026
#
# counts.csv columns (case-insensitive):
#   generator, n_valid, n_total
# Also accepted aliases:
#   generator_model -> generator
#   valid -> n_valid
#   total -> n_total
#
# Optional:
#   --pairwise   Run pairwise proportion tests (BH-adjusted)
#   --quiz_batch_tag TAG   When parsing --agg_log, read the tagged summary block
#                          (e.g., TAG=Jan2026 parses
#                          "[INFO] Questions passing judge filters (tagged manifests: Jan2026)")
#   -h, --help   Show help

usage <- function() {
  cat(paste(
    "Test whether validity rates differ by generator (binomial GLM).",
    "",
    "Required (choose one):",
    "  --counts_csv PATH   CSV with columns: generator, n_valid, n_total",
    "  --agg_log PATH      Text file containing aggregate_results.py validity summary",
    "",
    "Optional:",
    "  --pairwise          Pairwise tests via pairwise.prop.test (BH-adjusted)",
    "  --quiz_batch_tag TAG  When using --agg_log, parse the tagged summary block (e.g., Jan2026)",
    "  -h, --help          Show this help",
    "",
    "Examples:",
    "  Rscript r_analysis/validity_by_generator.R --counts_csv r_analysis/validity_counts.csv",
    "  python quizbench/aggregate_results.py --runs_root eval_results/quizbench/quizzes_Jan2026/runs --quiz_batch_tag Jan2026 > agg.txt",
    "  Rscript r_analysis/validity_by_generator.R --agg_log agg.txt --quiz_batch_tag Jan2026",
    "",
    sep = "\n"
  ))
}

parse_args <- function(args) {
  opts <- list(pairwise = FALSE)
  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]

    if (arg %in% c("-h", "--help")) {
      opts[["help"]] <- TRUE
      i <- i + 1
      next
    }
    if (arg == "--pairwise") {
      opts[["pairwise"]] <- TRUE
      i <- i + 1
      next
    }

    if (!grepl("^--", arg)) {
      stop(paste("Unrecognized argument:", arg))
    }

    if (grepl("=", arg)) {
      key <- sub("^--", "", sub("=.*", "", arg))
      val <- sub("^--[^=]+=", "", arg)
    } else {
      key <- sub("^--", "", arg)
      if (i == length(args)) {
        stop(paste("Missing value for", arg))
      }
      val <- args[[i + 1]]
      i <- i + 1
    }
    opts[[key]] <- val
    i <- i + 1
  }
  opts
}

normalize_counts <- function(df) {
  if (!is.data.frame(df) || nrow(df) == 0) {
    stop("No rows found in counts input.")
  }

  names(df) <- tolower(names(df))

  if (!("generator" %in% names(df))) {
    if ("generator_model" %in% names(df)) {
      df$generator <- df$generator_model
    } else {
      stop("Missing column 'generator' (or alias 'generator_model').")
    }
  }

  if (!("n_valid" %in% names(df))) {
    if ("valid" %in% names(df)) {
      df$n_valid <- df$valid
    } else {
      stop("Missing column 'n_valid' (or alias 'valid').")
    }
  }

  if (!("n_total" %in% names(df))) {
    if ("total" %in% names(df)) {
      df$n_total <- df$total
    } else {
      stop("Missing column 'n_total' (or alias 'total').")
    }
  }

  df <- df[, c("generator", "n_valid", "n_total")]

  df$generator <- trimws(as.character(df$generator))
  df$n_valid <- suppressWarnings(as.integer(df$n_valid))
  df$n_total <- suppressWarnings(as.integer(df$n_total))

  if (any(df$generator == "" | is.na(df$generator))) {
    stop("Blank/NA generator values found.")
  }
  if (any(is.na(df$n_valid) | is.na(df$n_total))) {
    stop("NA values found in n_valid/n_total after integer conversion.")
  }

  excluded_generators <- c("_EXAMPLE")
  to_exclude <- tolower(df$generator) %in% excluded_generators
  if (any(to_exclude)) {
    df <- df[!to_exclude, , drop = FALSE]
  }
  if (nrow(df) == 0) {
    stop("No generators remain after excluding '_EXAMPLE'.")
  }
  if (any(df$n_total <= 0)) {
    stop("All n_total values must be > 0.")
  }
  if (any(df$n_valid < 0 | df$n_valid > df$n_total)) {
    stop("All n_valid values must satisfy 0 <= n_valid <= n_total.")
  }

  df$n_invalid <- df$n_total - df$n_valid
  df$validity_rate <- df$n_valid / df$n_total
  df$generator <- as.factor(df$generator)

  df
}

read_counts_csv <- function(path) {
  if (is.null(path) || path == "") {
    stop("Missing --counts_csv PATH")
  }
  if (!file.exists(path)) {
    stop(paste("counts_csv not found:", path))
  }
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  normalize_counts(df)
}

read_counts_from_agg_log <- function(path, quiz_batch_tag = NULL) {
  if (is.null(path) || path == "") {
    stop("Missing --agg_log PATH")
  }
  if (!file.exists(path)) {
    stop(paste("agg_log not found:", path))
  }

  lines <- readLines(path, warn = FALSE)
  label <- if (!is.null(quiz_batch_tag) && quiz_batch_tag != "") {
    paste0("Questions passing judge filters (tagged manifests: ", quiz_batch_tag, ")")
  } else {
    "Questions passing judge filters"
  }
  header <- paste0("[INFO] ", label)
  start_idx <- grep(header, lines, fixed = TRUE)
  if (length(start_idx) == 0) {
    stop(paste(
      "Could not find validity summary header in log:",
      header,
      "Tip: run quizbench/aggregate_results.py with judge filtering enabled and capture stdout.",
      sep = "\n"
    ))
  }

  i <- start_idx[[1]] + 1
  gens <- character()
  n_valid <- integer()
  n_total <- integer()

  pattern <- "^\\s*(.+?):\\s*valid=([0-9]+)\\s*/\\s*total=([0-9]+)\\s*\\("
  while (i <= length(lines)) {
    line <- lines[[i]]
    if (trimws(line) == "") {
      break
    }

    m <- regmatches(line, regexec(pattern, line))[[1]]
    if (length(m) == 4) {
      gen <- trimws(m[[2]])
      if (gen != "TOTAL") {
        gens <- c(gens, gen)
        n_valid <- c(n_valid, as.integer(m[[3]]))
        n_total <- c(n_total, as.integer(m[[4]]))
      }
    }
    i <- i + 1
  }

  df <- data.frame(generator = gens, n_valid = n_valid, n_total = n_total,
                   stringsAsFactors = FALSE)
  normalize_counts(df)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0 || any(args %in% c("-h", "--help"))) {
  usage()
  quit(status = 0)
}

opts <- parse_args(args)
if (isTRUE(opts[["help"]])) {
  usage()
  quit(status = 0)
}

counts_csv <- opts[["counts_csv"]]
agg_log <- opts[["agg_log"]]
pairwise <- isTRUE(opts[["pairwise"]])
quiz_batch_tag <- opts[["quiz_batch_tag"]]
if ((is.null(quiz_batch_tag) || quiz_batch_tag == "") && !is.null(opts[["batch_tag"]])) {
  quiz_batch_tag <- opts[["batch_tag"]]
}

if ((is.null(counts_csv) || counts_csv == "") && (is.null(agg_log) || agg_log == "")) {
  usage()
  stop("Provide exactly one of --counts_csv PATH or --agg_log PATH")
}
if (!is.null(counts_csv) && counts_csv != "" && !is.null(agg_log) && agg_log != "") {
  usage()
  stop("Provide only one of --counts_csv PATH or --agg_log PATH (not both).")
}

gen_counts <- if (!is.null(counts_csv) && counts_csv != "") {
  read_counts_csv(counts_csv)
} else {
  read_counts_from_agg_log(agg_log, quiz_batch_tag = quiz_batch_tag)
}

gen_counts <- gen_counts[order(gen_counts$generator), ]

if (nlevels(gen_counts$generator) < 2) {
  stop("Need at least 2 generators for a between-generator validity test.")
}

cat("\nPer-generator validity counts:\n")
print(gen_counts[, c("generator", "n_valid", "n_total", "validity_rate")], row.names = FALSE)

m <- glm(
  cbind(n_valid, n_invalid) ~ generator,
  family = binomial,
  data = gen_counts
)

cat("\nGlobal test (do validity rates differ by generator?):\n")
print(anova(m, test = "Chisq"))

cat("\nModel summary:\n")
print(summary(m))

if (pairwise) {
  cat("\nPairwise proportion tests (BH-adjusted):\n")
  print(pairwise.prop.test(gen_counts$n_valid, gen_counts$n_total, p.adjust.method = "BH"))
}
