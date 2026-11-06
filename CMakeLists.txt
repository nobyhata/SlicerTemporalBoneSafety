cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerTemporalBoneSafety)

#-----------------------------------------------------------------------------
# Extension meta-information (URLs assume a future GitHub repo nobyhata/SlicerTemporalBoneSafety; update when published)
set(EXTENSION_HOMEPAGE "https://github.com/nobyhata/SlicerTemporalBoneSafety")
set(EXTENSION_CATEGORY "IGT")
set(EXTENSION_CONTRIBUTORS "Nobuhiko Hata (Brigham and Women's Hospital)")
set(EXTENSION_DESCRIPTION "Safety-margin map, virtual drilling and drill-tip distance monitoring for otologic temporal bone surgery.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/nobyhata/SlicerTemporalBoneSafety/main/TemporalBoneSafety/Resources/Icons/TemporalBoneSafety.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/nobyhata/SlicerTemporalBoneSafety/main/Screenshots/screenshot.png")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
add_subdirectory(TemporalBoneSafety)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
