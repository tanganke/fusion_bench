for script in \
    evaluate_gemma-2-9b.sh \
    evaluate_gemma-2-9b-it.sh \
    evaluate_Llama-3.1-8B.sh \
    evaluate_Llama-3.1-8B-Instruct.sh \
    evaluate_gemma-2-2b.sh \
    evaluate_gemma-2-2b-it.sh \
    evaluate_Llama-3.2-3B.sh \
    evaluate_Llama-3.2-3B-Instruct.sh; do
    echo "Running $script"
    bash $script
done
