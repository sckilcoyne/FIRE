mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
