#ifndef VANGERS_LIB_AVI_H
#define VANGERS_LIB_AVI_H

/*
 * Forwarding header to satisfy legacy includes of "avi.h" from sources
 * which expect the header to be available at the top-level lib include
 * directory. The real implementation lives in lib/xsound/avi.h; include
 * that file here so existing #include "avi.h" usages continue to work.
 *
 * This file intentionally does not introduce any overrides or fallbacks;
 * it simply forwards to the canonical header so the build uses the real
 * AVI interface.
 */

#include "xsound/avi.h"

#endif // VANGERS_LIB_AVI_H