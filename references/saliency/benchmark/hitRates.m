function [hitRate , falseAlarm] = hitRates(testMap,gtMap)
neg_gtMap = ~gtMap;
neg_testMap = ~testMap;

hitCount = sum(sum(testMap.*gtMap));
trueAvoidCount = sum(sum(neg_testMap.*neg_gtMap));
missCount = sum(sum(testMap.*neg_gtMap));
falseAvoidCount = sum(sum(neg_testMap.*gtMap));

falseAlarm = 1 - trueAvoidCount / (eps+trueAvoidCount + missCount);

hitRate = hitCount / (eps+ hitCount + falseAvoidCount);
end





