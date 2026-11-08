#include <iomanip>
#include <iostream>

#include <mocs/moga.h>
#include <mocs/params.h>
#include <mocs/realgene.h>

class ExposedMOGA : public MOGA {
public:
    explicit ExposedMOGA(Params& params) : MOGA(params) {}
    void rank(Population& pop) { assignFitness(pop); }
    void retain(Population& pop, Population& old) { addBest(pop, old); }
};

static Individual* make_individual(double tag, double first, double second) {
    Individual* item = new Individual;
    RealGene* marker = new RealGene(0.0, 100.0);
    marker->setValue(tag);
    item->getGenome().add(marker);
    item->getFitness().resize(2);
    item->getFitness()[0] = first;
    item->getFitness()[1] = second;
    item->setValue(1.0e300);
    return item;
}

static void emit_population(const Population& pop) {
    std::cout << "[";
    for (size_t i = 0; i < pop.size(); ++i) {
        const Individual* item = pop[i];
        const RealGene* marker = (const RealGene*)item->getGenome()[0];
        if (i) std::cout << ",";
        std::cout << "{\"tag\":" << marker->getValue()
                  << ",\"fitness\":[" << item->getFitness()[0] << ","
                  << item->getFitness()[1] << "]"
                  << ",\"value\":" << item->getValue() << "}";
    }
    std::cout << "]";
}

static void put(Params& params, const char* key, const char* value) {
    params.put(key, value);
}

int main() {
    std::cout << std::setprecision(17);

    {
        Params params;
        put(params, "parents", "3");
        put(params, "population", "3");
        put(params, "best", "0");
        put(params, "sharing", "0.1");
        ExposedMOGA algorithm(params);
        Population candidates;
        candidates.add(make_individual(0, 0.1, 0.2));
        candidates.add(make_individual(1, 0.2, 0.3));
        candidates.add(make_individual(2, 0.3, 0.4));
        candidates.add(make_individual(3, 0.4, 0.5));
        candidates.add(make_individual(4, 0.5, 0.6));
        algorithm.rank(candidates);
        std::cout << "{\"rank_unique\":{\"population\":";
        emit_population(candidates);
        std::cout << "}}\n";
    }

    {
        Params params;
        put(params, "parents", "3");
        put(params, "population", "3");
        put(params, "best", "1");
        put(params, "sharing", "0.1");
        ExposedMOGA algorithm(params);
        Population offspring;
        offspring.add(make_individual(20, 0.1, 0.2));
        offspring.add(make_individual(21, 0.2, 0.3));
        offspring.add(make_individual(22, 0.3, 0.4));
        offspring.add(make_individual(23, 0.4, 0.5));
        algorithm.rank(offspring);

        Population old;
        old.add(make_individual(10, 0.1, 0.8));
        old.add(make_individual(11, 0.2, 0.6));
        old.add(make_individual(12, 0.2, 0.6));
        old.add(make_individual(13, 0.4, 0.9));
        algorithm.retain(offspring, old);
        std::cout << "{\"front_dedup_then_best_duplicate\":{\"result\":";
        emit_population(offspring);
        std::cout << ",\"old_after_best_sort\":";
        emit_population(old);
        std::cout << "}}\n";
    }

    {
        Params params;
        put(params, "parents", "2");
        put(params, "population", "2");
        put(params, "best", "3");
        put(params, "sharing", "0.1");
        ExposedMOGA algorithm(params);
        Population offspring;
        offspring.add(make_individual(30, 0.1, 0.2));
        offspring.add(make_individual(31, 0.2, 0.3));
        offspring.add(make_individual(32, 0.3, 0.4));
        algorithm.rank(offspring);
        Population old;
        old.add(make_individual(40, 0.3, 0.2));
        old.add(make_individual(41, 0.1, 0.4));
        old.add(make_individual(42, 0.2, 0.3));
        algorithm.retain(offspring, old);
        std::cout << "{\"best_equals_old_size_skips_front_block\":{\"result\":";
        emit_population(offspring);
        std::cout << ",\"old_order_unchanged\":";
        emit_population(old);
        std::cout << "}}\n";
    }
    return 0;
}
