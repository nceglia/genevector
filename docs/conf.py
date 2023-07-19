import os
import sys
sys.path.insert(0, os.path.abspath('.'))

project = 'GeneVector'
copyright = '2023, Memorial Sloan Kettering Cancer Center'
author = 'Nicholas Ceglia'

version = ''
release = '1.0.0'

extensions = []

templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'

html_logo = "../logo.png"

language = None

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

htmlhelp_basename = 'MyProjectdoc'