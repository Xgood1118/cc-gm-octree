import argparse
import octree as m


def main():
    ap = argparse.ArgumentParser(description="八叉树构建与区域查询")
        ap.add_argument("--radius", type=float, default=2.5)
        ap.add_argument("--out", required=True)
    a = ap.parse_args()
        print("TODO")


if __name__ == "__main__":
    main()
