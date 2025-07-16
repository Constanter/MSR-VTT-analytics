import argparse
from pathlib import Path
from extract_embeddings import extract_embeddings
from cluster_embeddings import cluster_embeddings
from generate_descriptions import generate_cluster_descriptions

def main():
    parser = argparse.ArgumentParser(description='MSR-VTT Video Clustering Pipeline')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory with videos')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--subset_size', type=int, default=None, help='Number of videos to process')
    parser.add_argument('--n_clusters', type=int, default=20, help='Number of clusters')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Шаг 1: Извлечение эмбеддингов
    print("Step 1/3: Extracting embeddings...")
    extract_embeddings(
        video_dir=Path(args.video_dir).absolute(),
        output_dir=output_dir,
        subset_size=args.subset_size
    )

    
    # Шаг 2: Кластеризация
    print("\nStep 2/3: Clustering embeddings...")
    cluster_df = cluster_embeddings(
        embeddings_file=output_dir / 'embeddings.json',
        output_dir=output_dir,
        n_clusters=args.n_clusters
    )
    
    # Шаг 3: Генерация описаний
    print("\nStep 3/3: Generating cluster descriptions...")    
    generate_cluster_descriptions(
        cluster_mapping=output_dir / 'video_cluster_mapping.csv',
        video_description_file=args.video_description_file,
        output_dir=output_dir
    )
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()