import networkx as nx
import matplotlib.pyplot as plt


def load_csv(fname: str):
    """
    plot the number of citations in each year for the MAG dataset
    """
    f = open(fname, "r")
    lines = list(f.readlines())
    f.close()

    years = []
    cites = []
    for i in range(len(lines)):
        if i == 0:
            continue
        line = lines[i]
        line = line.split(",")
        try:
            year = int(line[0])
        except:
            continue
        num_citations = int(line[1])
        years.append(year)
        cites.append(num_citations)

    plt.plot(years, cites, color="#e34a33")
    plt.xlabel("Year")
    plt.ylabel("Paper Count")
    plt.savefig("paper_count.pdf")
    plt.close()


if __name__ == "__main__":
    load_csv("paper_year.txt")
