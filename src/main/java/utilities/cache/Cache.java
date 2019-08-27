package utilities.cache;

import java.util.HashMap;
import java.util.function.Supplier;

public class Cache<A, B, C> {
    private final HashMap<A, HashMap<B, C>> cache = new HashMap<>();

    public C getAndPut(A firstKey, B secondKey, Supplier<C> supplier) {
        C result = get(firstKey, secondKey);
        if(result == null) {
            result = supplier.get();
        }
        put(firstKey, secondKey, result);
        return result;
    }

    public C get(A firstKey, B secondKey) {
        C result = null;
        HashMap<B, C> subCache = cache.get(firstKey);
        if(subCache != null) {
            result = subCache.get(secondKey);
        }
        return result;
    }

    public void put(A firstKey, B secondkey, C value) {
        HashMap<B, C> subCache = cache.computeIfAbsent(firstKey, k -> new HashMap<>());
        subCache.put(secondkey, value);
    }

    public void clear() {
        cache.clear();
    }

    public void remove(A firstKey, B secondKey) {
        HashMap<B, C> subCache = cache.get(firstKey);
        if(subCache != null) {
            subCache.remove(secondKey);
            if(subCache.isEmpty()) {
                cache.remove(firstKey);
            }
        }
    }
}