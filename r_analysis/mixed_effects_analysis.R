#!/usr/bin/env Rscript
# Mixed-effects logistic regression for item-level model comparisons.
# Usage:
#   Rscript dev/mixed_effects_analysis.R --input path/to/item_level.csv \
#     [--outdir R_results/glmm_analysis_YYYYMMDDTHHMMSS] \
#     [--adjust holm] [--reference MODEL_NAME]
#
# Required columns in the input CSV (long format):
#   model, quiz_id, question_id, correct
#
# Example input row:
#   gpt-4o,1,12,1

if (!requireNamespace("lme4", quietly = TRUE)) {
  stop("Package 'lme4' is required. Install with install.packages('lme4').")
}
if (!requireNamespace("emmeans", quietly = TRUE)) {
  stop("Package 'emmeans' is required. Install with install.packages('emmeans').")
}

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
})

usage <- function() {
  cat(paste(
    "Mixed-effects logistic regression for item-level model comparisons.",
    "",
    "Required:",
    "  --input PATH             CSV with columns: model, quiz_id, question_id, correct",
    "",
    "Optional:",
    "  --outdir PATH            Output directory (default: R_results/glmm_analysis_<timestamp>)",
    "  --adjust METHOD          Multiple-comparisons adjustment for pairwise tests (default: holm)",
    "  --reference MODEL_NAME   Reference model level for fixed effects",
    "  -h, --help               Show this help",
    "",
    sep = "\n"
  ))
}

parse_args <- function(args) {
  opts <- list()
  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]
    if (grepl("^--", arg)) {
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
    } else if (arg %in% c("-h", "--help")) {
      opts[["help"]] <- "true"
    } else {
      stop(paste("Unrecognized argument:", arg))
    }
    i <- i + 1
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

input_path <- opts[["input"]]
if (is.null(input_path) || input_path == "") {
  stop("Missing required --input PATH")
}

timestamp <- format(Sys.time(), "%Y%m%dT%H%M%S")
outdir <- opts[["outdir"]]
if (is.null(outdir) || outdir == "") {
  outdir <- file.path("R_results", paste0("glmm_analysis_", timestamp))
}

adjust_method <- opts[["adjust"]]
if (is.null(adjust_method) || adjust_method == "") {
  adjust_method <- "holm"
}

ref_model <- opts[["reference"]]
if (!file.exists(input_path)) {
  stop(paste("Input file not found:", input_path))
}

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(input_path, stringsAsFactors = FALSE)
required_cols <- c("model", "quiz_id", "question_id", "correct")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

if (anyNA(df[, required_cols])) {
  stop("Missing values detected in required columns; please clean before analysis.")
}

df$model <- as.factor(df$model)
df$quiz_id <- as.factor(df$quiz_id)
df$question_id <- as.factor(df$question_id)

if (is.logical(df$correct)) {
  df$correct <- as.integer(df$correct)
} else if (is.character(df$correct)) {
  vals <- tolower(trimws(df$correct))
  is_one <- vals %in% c("1", "true", "t", "yes", "y")
  is_zero <- vals %in% c("0", "false", "f", "no", "n")
  if (any(!(is_one | is_zero))) {
    stop("Column 'correct' must be binary (0/1 or true/false).")
  }
  df$correct <- ifelse(is_one, 1L, 0L)
}

if (!is.numeric(df$correct)) {
  stop("Column 'correct' must be numeric, logical, or character values of 0/1.")
}

if (any(!(df$correct %in% c(0, 1)))) {
  stop("Column 'correct' must be binary (0/1).")
}

if (!is.null(ref_model) && ref_model != "") {
  if (!(ref_model %in% levels(df$model))) {
    stop(paste("Reference model not found in data:", ref_model))
  }
  df$model <- relevel(df$model, ref = ref_model)
}

df$question_in_quiz <- interaction(df$quiz_id, df$question_id, drop = TRUE)

dup_counts <- aggregate(correct ~ model + quiz_id + question_id,
                        data = df, FUN = length)
dup_issues <- dup_counts[dup_counts$correct != 1, ]

pair_counts <- aggregate(correct ~ quiz_id + question_id,
                         data = df, FUN = length)
expected_models <- length(levels(df$model))
pair_issues <- pair_counts[pair_counts$correct != expected_models, ]

obs_acc <- aggregate(correct ~ model, data = df, FUN = mean)
names(obs_acc)[2] <- "observed_accuracy"

data_summary <- c(
  paste("rows:", nrow(df)),
  paste("models:", length(levels(df$model))),
  paste("quizzes:", length(levels(df$quiz_id))),
  paste("questions:", length(levels(df$question_in_quiz))),
  paste("expected_models_per_item:", expected_models),
  paste("pairing_issues:", nrow(pair_issues)),
  paste("duplicate_responses:", nrow(dup_issues))
)

writeLines(data_summary, con = file.path(outdir, "data_summary.txt"))
write.csv(obs_acc, file.path(outdir, "observed_accuracy.csv"), row.names = FALSE)
if (nrow(pair_issues) > 0) {
  write.csv(pair_issues, file.path(outdir, "pairing_issues.csv"), row.names = FALSE)
  warning("Pairing issues detected; see pairing_issues.csv for details.")
}
if (nrow(dup_issues) > 0) {
  write.csv(dup_issues, file.path(outdir, "duplicate_responses.csv"), row.names = FALSE)
  warning("Duplicate responses detected; see duplicate_responses.csv for details.")
}

control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

full_model <- glmer(
  correct ~ model + (1 | quiz_id) + (1 | quiz_id:question_id),
  data = df,
  family = binomial,
  control = control
)

null_model <- glmer(
  correct ~ 1 + (1 | quiz_id) + (1 | quiz_id:question_id),
  data = df,
  family = binomial,
  control = control
)

lrt <- anova(null_model, full_model, test = "Chisq")
write.csv(as.data.frame(lrt), file.path(outdir, "omnibus_lrt.csv"), row.names = FALSE)

if (isSingular(full_model)) {
  warning("Full model is singular; random-effect variance may be near zero.")
}
conv_msgs <- full_model@optinfo$conv$lme4$messages
if (!is.null(conv_msgs)) {
  warning(paste("Convergence warnings:", paste(conv_msgs, collapse = " | ")))
}

emm <- emmeans(full_model, ~ model)
emm_prob <- emmeans(full_model, ~ model, type = "response")
emm_prob_summary <- summary(emm_prob, infer = TRUE)
write.csv(as.data.frame(emm_prob_summary),
          file.path(outdir, "model_predicted_accuracy.csv"),
          row.names = FALSE)

pair_logit <- pairs(emm, adjust = adjust_method)
pair_logit_summary <- summary(pair_logit, infer = TRUE)
write.csv(as.data.frame(pair_logit_summary),
          file.path(outdir, "pairwise_logit_contrasts.csv"),
          row.names = FALSE)

pair_or_summary <- summary(pair_logit, infer = TRUE, type = "response")
write.csv(as.data.frame(pair_or_summary),
          file.path(outdir, "pairwise_odds_ratios.csv"),
          row.names = FALSE)

pair_diff <- contrast(emm_prob, method = "pairwise")
pair_diff_summary <- summary(pair_diff, infer = TRUE, adjust = adjust_method)
write.csv(as.data.frame(pair_diff_summary),
          file.path(outdir, "pairwise_accuracy_diff.csv"),
          row.names = FALSE)

cat("Omnibus LRT (model effect):\n")
print(lrt)
cat("\nPredicted accuracy by model (marginal means):\n")
print(emm_prob_summary)
cat("\nPairwise comparisons (log-odds differences, adjusted):\n")
print(pair_logit_summary)
cat("\nPairwise comparisons (odds ratios, adjusted):\n")
print(pair_or_summary)
cat("\nPairwise comparisons (accuracy differences, adjusted; approx on response scale):\n")
print(pair_diff_summary)
cat("\nOutputs written to:", outdir, "\n")
