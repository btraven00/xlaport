# Tests that execute a kernel need the PJRT plugin. pjrt only downloads it
# when asked, and refuses in non-interactive sessions (R CMD check,
# r-universe) unless PJRT_INSTALL=1. Run those tests only when the plugin
# is already there or a download/local path has been allowed.
skip_if_no_pjrt <- function() {
  skip_if_not_installed("pjrt")
  platform <- Sys.getenv("PJRT_PLATFORM", "cpu")
  if (pjrt::plugins_downloaded(platform) ||
      Sys.getenv("PJRT_INSTALL") == "1" ||
      nzchar(Sys.getenv(paste0("PJRT_PLUGIN_PATH_", toupper(platform))))) {
    return(invisible(TRUE))
  }
  skip(sprintf("PJRT %s plugin not downloaded; set PJRT_INSTALL=1 to fetch it", platform))
}
