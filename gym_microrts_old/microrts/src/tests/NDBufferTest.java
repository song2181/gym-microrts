package tests;

import java.nio.IntBuffer;
import java.util.Arrays;

import util.NDBuffer;

public class NDBufferTest {

    private static int sumBuffer(NDBuffer buffer) {
        final IntBuffer buff = buffer.getBuffer();
        buff.rewind();
        int total = 0;
        while(buff.hasRemaining()) {
            total += buff.get();
        }
        return total;
    }

    public static void main(String[] args) {
        final int capacity = 2 * 2 * 3;
        final IntBuffer buff = IntBuffer.allocate(capacity);
        final NDBuffer ndbuff = new NDBuffer(buff, new int[]{2, 2, 3});
        
        assert sumBuffer(ndbuff) == 0;

        ndbuff.set(new int[]{0, 1, 0}, 100);
        ndbuff.set(new int[]{1, 1, 0}, 10);
        ndbuff.set(new int[]{1, 1, 1}, 20);
        ndbuff.set(new int[]{1, 1, 2}, 30);

        System.out.println(Arrays.toString(ndbuff.getBuffer().array()));
        assert sumBuffer(ndbuff) == 160;

        ndbuff.resetSegment(new int[]{1, 1});

        System.out.println(Arrays.toString(ndbuff.getBuffer().array()));
        assert sumBuffer(ndbuff) == 100;

        final int[] dest = new int[3];
        ndbuff.getSegment(new int[]{0, 1}, dest);
        System.out.println(Arrays.toString(dest));
    }
    
}