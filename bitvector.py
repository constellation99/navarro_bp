import math
import unittest
from bitarray import bitarray


class BitVector:
    def __init__(self, bitstring, superblock_size=512, subblock_size=16):
        """
        bitstring: A string of '0'/'1', or an iterable of 0/1 bits.
        superblock_size, subblock_size: the sizes for two-level rank.
        """
        # Convert input into a bitarray
        self.B = bitarray(bitstring)
        self.n = len(self.B)
        self.superblock_size = superblock_size
        self.subblock_size = subblock_size

        # We'll figure out how many superblocks we have
        self.num_superblocks = (self.n + superblock_size - 1) // superblock_size

        # Top-level arrays for rank1
        # rank1_superblock[sb] = total # of 1s in [0 .. sb_start_of_superblock-1]
        self.rank1_superblock = [0]*self.num_superblocks

        # rank1_subblock[sb] = a list of length (#subblocks in superblock sb),
        # each entry giving # of 1s from the start of that superblock up to the *end* of that subblock.
        # Example: rank1_subblock[sb][k] = # of 1 bits in the sub-blocks 0..k (within superblock sb).
        self.rank1_subblock = [[] for _ in range(self.num_superblocks)]

        # We'll store total number of 1 bits as well
        self.total_ones = 0

        # We do the same structure for rank10
        self.rank10_superblock = [0]*self.num_superblocks
        self.rank10_subblock = [[] for _ in range(self.num_superblocks)]
        self.total_ten_patterns = 0

        # Precompute everything
        self.precompute_rank_structures()

    def precompute_rank_structures(self):
        """
        Build the 2D rank data (rank1_superblock, rank1_subblock)
        and similarly for rank10.
        """
        cumulative_ones = 0
        cumulative_ten = 0

        sb_index = 0  # index of current superblock
        pos = 0       # index in the global bitvector

        while pos < self.n:
            # The start of the current superblock
            sb_start = pos
            sb_end   = min(sb_start + self.superblock_size, self.n)

            # Store how many 1s we have so far for the superblock boundary
            self.rank1_superblock[sb_index] = cumulative_ones
            self.rank10_superblock[sb_index] = cumulative_ten

            # Now we iterate sub-block by sub-block within this superblock
            subblock_count_ones = 0
            subblock_count_ten  = 0
            subblock_list_ones = []
            subblock_list_ten  = []

            local_pos = sb_start
            sub_idx   = 0

            while local_pos < sb_end:
                # Start of the subblock
                sub_start = local_pos
                sub_end   = min(sub_start + self.subblock_size, sb_end)

                # We count how many 1 bits and how many '10' patterns in [sub_start .. sub_end-1]
                local_ones = 0
                local_ten  = 0
                prev_bit   = self.B[sub_start - 1] if sub_start > 0 else None

                for p in range(sub_start, sub_end):
                    bit_val = self.B[p]
                    local_ones += bit_val
                    # Check for '10' pattern: we need the previous bit to be 1 and current bit to be 0
                    if p > 0:
                        if self.B[p-1] == 1 and self.B[p] == 0:
                            local_ten += 1

                # Update subblock_count
                subblock_count_ones += local_ones
                subblock_count_ten  += local_ten

                # Now we store the *cumulative* (within superblock) after sub_idx sub-blocks
                subblock_list_ones.append(subblock_count_ones)
                subblock_list_ten.append(subblock_count_ten)

                local_pos = sub_end
                sub_idx  += 1

            # Done with all subblocks in this superblock
            # Now add subblock_list_ones to rank1_subblock[sb_index]
            self.rank1_subblock[sb_index] = subblock_list_ones
            self.rank10_subblock[sb_index] = subblock_list_ten

            # Update the global counts
            local_ones_sum = subblock_count_ones
            local_ten_sum  = subblock_count_ten

            cumulative_ones += local_ones_sum
            cumulative_ten  += local_ten_sum

            # Move to next superblock
            sb_index += 1
            pos = sb_end

        # Finally store the totals
        self.total_ones = cumulative_ones
        self.total_ten_patterns = cumulative_ten

    def rank1(self, i):
        """
        Return the number of 1 bits in B[0..i], inclusive.
        If i < 0, return 0. If i >= n, return total_ones.
        """
        if i < 0:
            return 0
        if i >= self.n:
            return self.total_ones

        # Which superblock is i in?
        sb_idx = i // self.superblock_size
        # This is the # of 1 bits before this superblock
        rank_val = self.rank1_superblock[sb_idx]

        # Now inside this superblock, figure out which subblock
        offset_within_sb = i % self.superblock_size
        sub_idx = offset_within_sb // self.subblock_size

        # Add the subblock count (cumulative within the superblock)
        if sub_idx > 0:
            rank_val += self.rank1_subblock[sb_idx][sub_idx - 1]
        # Now we do a small linear scan for the remainder of the subblock
        sub_start = (sb_idx * self.superblock_size) + (sub_idx * self.subblock_size)
        for pos in range(sub_start, i + 1):
            rank_val += self.B[pos]

        return rank_val

    def rank0(self, i):
        """
        Return the number of 0 bits in B[0..i], inclusive.
        If i < 0, return 0. If i >= n, return total number of 0s.
        """
        if i < 0:
            return 0
        if i >= self.n:
            return self.n - self.total_ones
        # Can be computed as (i+1) - rank1(i)
        # Total bits counted is (i+1), then subtract the number of 1s.
        return (i + 1) - self.rank1(i)

    def rank10(self, i):
        """
        Return the number of '10' patterns in B[0..i], inclusive of i if i>0.
        If i < 1, return 0. If i >= n, return total_ten_patterns.
        """
        if i < 1:
            return 0
        if i >= self.n:
            return self.total_ten_patterns

        sb_idx = i // self.superblock_size
        rank_val = self.rank10_superblock[sb_idx]

        offset_within_sb = i % self.superblock_size
        sub_idx = offset_within_sb // self.subblock_size

        if sub_idx > 0:
            rank_val += self.rank10_subblock[sb_idx][sub_idx - 1]

        sub_start = (sb_idx * self.superblock_size) + (sub_idx * self.subblock_size)
        # We only start checking '10' from max(sub_start, 1) 
        # because we need to look at pairs (p-1, p).
        scan_start = max(sub_start, 1)
        for pos in range(scan_start, i+1):
            if self.B[pos-1] == 1 and self.B[pos] == 0:
                rank_val += 1

        return rank_val

    def select1(self, k):
        """
        Return the index i such that rank1(i) = k (1-based for '1's).
        If multiple, returns the first occurrence. Raises ValueError if k out of range.
        """
        if k <= 0 or k > self.total_ones:
            raise ValueError("k is out of bounds for select1")

        # 1) Binary search over superblocks
        left, right = 0, self.num_superblocks - 1
        while left <= right:
            mid = (left + right) // 2
            if self.rank1_superblock[mid] < k:
                left = mid + 1
            else:
                right = mid - 1
        sb_idx = right
        if sb_idx < 0:
            sb_idx = 0

        # Adjust k relative to the superblock boundary
        k -= self.rank1_superblock[sb_idx]

        # 2) Linear search subblocks in that superblock
        # subblock array is self.rank1_subblock[sb_idx]
        sub_arr = self.rank1_subblock[sb_idx]
        sb_start = sb_idx * self.superblock_size

        # We do a linear pass over subblocks
        prev_count = 0
        sub_start_idx = 0  # in bits
        for s_i, cval in enumerate(sub_arr):
            block_count = cval - prev_count
            if block_count >= k:
                # The k-th '1' is inside this subblock
                # we do a small linear scan
                block_bit_start = sb_start + s_i*self.subblock_size
                count_local = 0
                for bitpos in range(block_bit_start, min(block_bit_start+self.subblock_size, self.n)):
                    if self.B[bitpos] == 1:
                        count_local += 1
                        if count_local == k:
                            return bitpos
                raise ValueError("select1: Inconsistent counts in subblock scanning.")
            else:
                k -= block_count
            prev_count = cval

        # If not found in subblocks, might be in partial tail (if subblock_size doesn’t align exactly)
        # But typically sub_arr covers entire superblock, so we do a final linear pass if needed
        tail_start = sb_start + len(sub_arr)*self.subblock_size
        count_local = 0
        for bitpos in range(tail_start, min(sb_start+self.superblock_size, self.n)):
            if self.B[bitpos] == 1:
                count_local += 1
                if count_local == k:
                    return bitpos

        raise ValueError("select1: k out of bounds, not found in superblock tail.")

    def select0(self, k):
        """
        Returns the index i such that rank0(i) = k (1-based for zeros),
        computing zeros on the fly from rank1.
        Does not use extra precomputed zero structures.
        """
        total_zeros = self.n - self.total_ones
        if k <= 0 or k > total_zeros:
            raise ValueError("k is out of bounds for select0")
        # Compute cumulative zeros before each superblock on the fly:
        def cumulative_zeros(sb_idx):
            sb_start = sb_idx * self.superblock_size
            return sb_start - self.rank1_superblock[sb_idx]

        left, right = 0, self.num_superblocks - 1
        while left <= right:
            mid = (left + right) // 2
            zeros_before = cumulative_zeros(mid)
            sb_start = mid * self.superblock_size
            sb_end = min(sb_start + self.superblock_size, self.n)
            ones_in_sb = self.rank1_subblock[mid][-1] if self.rank1_subblock[mid] else 0
            zeros_in_sb = (sb_end - sb_start) - ones_in_sb
            if zeros_before + zeros_in_sb < k:
                left = mid + 1
            else:
                right = mid - 1
        sb_idx = left  # Superblock containing the kth zero.
        k -= cumulative_zeros(sb_idx)
        sb_start = sb_idx * self.superblock_size
        sb_end = min(sb_start + self.superblock_size, self.n)
        # Scan subblocks in the superblock; derive zeros count from ones count.
        sub_arr = self.rank1_subblock[sb_idx]
        local_zero_count = 0
        sub_idx = 0
        while sb_start + sub_idx * self.subblock_size < sb_end:
            subblock_start = sb_start + sub_idx * self.subblock_size
            subblock_end = min(subblock_start + self.subblock_size, sb_end)
            if sub_idx < len(sub_arr):
                ones_in_sub = sub_arr[sub_idx] - (sub_arr[sub_idx - 1] if sub_idx > 0 else 0)
            else:
                ones_in_sub = 0
            zeros_in_sub = (subblock_end - subblock_start) - ones_in_sub
            if local_zero_count + zeros_in_sub >= k:
                count = local_zero_count
                for pos in range(subblock_start, subblock_end):
                    if self.B[pos] == 0:
                        count += 1
                        if count == k:
                            return pos
                raise ValueError("select0: Inconsistent counts in subblock scanning.")
            else:
                local_zero_count += zeros_in_sub
            sub_idx += 1
        # Scan tail if necessary.
        tail_start = sb_start + sub_idx * self.subblock_size
        for pos in range(tail_start, sb_end):
            if self.B[pos] == 0:
                local_zero_count += 1
                if local_zero_count == k:
                    return pos
        raise ValueError("select0: k out of bounds, not found in superblock tail.")

    # def select0_precomputed(self, k):
    #     """
    #     Returns the index i such that rank0(i) = k (1-based for zeros),
    #     using precomputed arrays for zeros.
    #     """
    #     total_zeros = self.n - self.total_ones
    #     if k <= 0 or k > total_zeros:
    #         raise ValueError("k is out of bounds for select0_precomputed")
    #     left, right = 0, self.num_superblocks - 1
    #     while left <= right:
    #         mid = (left + right) // 2
    #         if self.rank0_superblock[mid] < k:
    #             left = mid + 1
    #         else:
    #             right = mid - 1
    #     sb_idx = right if right >= 0 else 0
    #     k -= self.rank0_superblock[sb_idx]
    #     sb_start = sb_idx * self.superblock_size
    #     prev_count = 0
    #     for s_i, cval in enumerate(self.rank0_subblock[sb_idx]):
    #         block_count = cval - prev_count
    #         if block_count >= k:
    #             block_bit_start = sb_start + s_i * self.subblock_size
    #             count_local = 0
    #             for pos in range(block_bit_start, min(block_bit_start + self.subblock_size, self.n)):
    #                 if self.B[pos] == 0:
    #                     count_local += 1
    #                     if count_local == k:
    #                         return pos
    #             raise ValueError("select0_precomputed: Inconsistent counts in subblock scanning.")
    #         else:
    #             k -= block_count
    #         prev_count = cval
    #     tail_start = sb_start + len(self.rank0_subblock[sb_idx]) * self.subblock_size
    #     count_local = 0
    #     for pos in range(tail_start, min(sb_start + self.superblock_size, self.n)):
    #         if self.B[pos] == 0:
    #             count_local += 1
    #             if count_local == k:
    #                 return pos
    #     raise ValueError("select0_precomputed: k out of bounds, not found in superblock tail.")

    def select10(self, k):
        """
        Return the index i such that rank10(i+1) = k,
        i.e. the position i of the '1' in the '10' pattern. 
        Raises ValueError if k out of range.
        """
        if k <= 0 or k > self.total_ten_patterns:
            raise ValueError("k is out of bounds for select10")

        # 1) Binary search over superblocks
        left, right = 0, self.num_superblocks - 1
        while left <= right:
            mid = (left + right) // 2
            if self.rank10_superblock[mid] < k:
                left = mid + 1
            else:
                right = mid - 1
        sb_idx = right
        if sb_idx < 0:
            sb_idx = 0

        # Adjust k relative to the superblock boundary
        k -= self.rank10_superblock[sb_idx]

        # 2) Linear search subblocks in that superblock
        sub_arr = self.rank10_subblock[sb_idx]
        sb_start = sb_idx * self.superblock_size

        prev_count = 0
        for s_i, cval in enumerate(sub_arr):
            block_count = cval - prev_count
            if block_count >= k:
                # small linear scan
                block_bit_start = sb_start + s_i*self.subblock_size
                # we scan from max(block_bit_start,1) to avoid negative indexing
                for bitpos in range(max(block_bit_start,1),
                                    min(block_bit_start+self.subblock_size, self.n)):
                    if self.B[bitpos-1] == 1 and self.B[bitpos] == 0:
                        k -= 1
                        if k == 0:
                            return bitpos-1
                raise ValueError("select10: Inconsistent counts in subblock scanning.")
            else:
                k -= block_count
            prev_count = cval

        # Accounts for remainder beyond the last subblock boundary
        tail_start = sb_start + len(sub_arr)*self.subblock_size
        for bitpos in range(max(tail_start,1), min(sb_start+self.superblock_size, self.n)):
            if self.B[bitpos-1] == 1 and self.B[bitpos] == 0:
                k -= 1
                if k == 0:
                    return bitpos-1

        raise ValueError("select10: k out of bounds, not found in superblock tail.")


class TestBitVector(unittest.TestCase):
    def test_basic_rank1_rank10_operations(self):
        """Test Case 1: Basic rank1, rank10, rank0, and select operations on '1100101'"""
        print("Test Case 1: Basic rank1, rank10, rank0, and select operations on '1100101'")
        bv = BitVector("1100101")

        # rank1 tests
        self.assertEqual(bv.rank1(0), 1, "rank1(0)")
        self.assertEqual(bv.rank1(3), 2, "rank1(3)")
        self.assertEqual(bv.rank1(6), 4, "rank1(6)")

        # select1 tests
        self.assertEqual(bv.select1(1), 0, "select1(1)")
        self.assertEqual(bv.select1(2), 1, "select1(2)")
        self.assertEqual(bv.select1(4), 6, "select1(4)")

        # rank10 tests
        self.assertEqual(bv.rank10(0), 0, "rank10(0)")
        self.assertEqual(bv.rank10(4), 1, "rank10(4)")
        self.assertEqual(bv.rank10(6), 2, "rank10(6)")

        # select10 tests
        self.assertEqual(bv.select10(1), 1, "select10(1)")
        self.assertEqual(bv.select10(2), 4, "select10(2)")

        # rank0 tests:
        # For "1100101" the bits are:
        # index:  0 1 2 3 4 5 6
        # bits:   1 1 0 0 1 0 1
        # So, rank0(0)=0, rank0(2)=1, rank0(3)=2, etc.
        self.assertEqual(bv.rank0(0), 0, "rank0(0)")
        self.assertEqual(bv.rank0(2), 1, "rank0(2)")
        self.assertEqual(bv.rank0(3), 2, "rank0(3)")

        # select0 tests:
        # Zeros are at indices 2, 3, and 5.
        # Testing the method without precomputed zero structures.
        self.assertEqual(bv.select0(1), 2, "select0(1)")
        self.assertEqual(bv.select0(2), 3, "select0(2)")
        self.assertEqual(bv.select0(3), 5, "select0(3)")


    def test_empty_bitvector(self):
        """Test Case 2: Edge Case with empty bitstring"""
        print("Test Case 2: Edge Case with empty bitstring")
        bv_empty = BitVector("")
        self.assertEqual(bv_empty.rank1(0), 0, "rank1 on empty bitvector")
        self.assertEqual(bv_empty.rank10(0), 0, "rank10 on empty bitvector")
        self.assertEqual(bv_empty.rank0(0), 0, "rank0 on empty bitvector")
        with self.assertRaises(ValueError, msg="select1(1) should raise ValueError for empty bitvector"):
            bv_empty.select1(1)
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for empty bitvector"):
            bv_empty.select10(1)
        with self.assertRaises(ValueError, msg="select0(1) should raise ValueError for empty bitvector"):
            bv_empty.select0(1)

        print("Empty bitvector operations passed.\n")

    def test_all_ones(self):
        """Test Case 3: BitVector with only '1's"""
        print("Test Case 3: BitVector with only '1's")
        bv_ones = BitVector("111111")
        self.assertEqual(bv_ones.rank1(5), 6, "rank1(5) - All Ones")
        self.assertEqual(bv_ones.select1(4), 3, "select1(4) - All Ones")
        self.assertEqual(bv_ones.rank10(5), 0, "rank10(5) - All Ones")
        # For an all-ones bitvector, rank0 returns 0 and select0 should raise an error.
        self.assertEqual(bv_ones.rank0(5), 0, "rank0(5) - All Ones")
        with self.assertRaises(ValueError, msg="select0(1) should raise ValueError for all-ones bitvector"):
            bv_ones.select0(1)
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for all-ones bitvector"):
            bv_ones.select10(1)
        print("All-ones bitvector operations passed.\n")

    def test_all_zeros(self):
        """Test Case 4: BitVector with only '0's"""
        print("Test Case 4: BitVector with only '0's")
        bv_zeros = BitVector("000000")
        self.assertEqual(bv_zeros.rank1(5), 0, "rank1(5) - All Zeros")
        self.assertEqual(bv_zeros.rank10(5), 0, "rank10(5) - All Zeros")
        # For all zeros, rank0 should equal the position+1 (up to total length).
        self.assertEqual(bv_zeros.rank0(5), 6, "rank0(5) - All Zeros")
        with self.assertRaises(ValueError, msg="select1(1) should raise ValueError for all-zeros bitvector"):
            bv_zeros.select1(1)
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for all-zeros bitvector"):
            bv_zeros.select10(1)
        # select0 should return positions: 0, 1, 2, etc.
        self.assertEqual(bv_zeros.select0(1), 0, "select0(1) - All Zeros")
        self.assertEqual(bv_zeros.select0(3), 2, "select0(3) - All Zeros")
        print("All-zeros bitvector operations passed.\n")

    def test_alternating_bits(self):
        """Test Case 5: Complex pattern '101010'"""
        print("Test Case 5: Complex pattern '101010'")
        bv_pattern = BitVector("101010")
        self.assertEqual(bv_pattern.rank1(5), 3, "rank1(5) - Alternating")
        self.assertEqual(bv_pattern.rank10(5), 3, "rank10(5) - Alternating")
        self.assertEqual(bv_pattern.select1(2), 2, "select1(2) - Alternating")
        self.assertEqual(bv_pattern.select10(1), 0, "select10(1) - Alternating")
        self.assertEqual(bv_pattern.select10(2), 2, "select10(2) - Alternating")
        # Test rank0 for "101010": bits are 1,0,1,0,1,0 so rank0(5) should be 3.
        self.assertEqual(bv_pattern.rank0(5), 3, "rank0(5) - Alternating")
        # Zeros occur at positions 1, 3, and 5.
        self.assertEqual(bv_pattern.select0(1), 1, "select0(1) - Alternating")
        self.assertEqual(bv_pattern.select0(2), 3, "select0(2) - Alternating")
        self.assertEqual(bv_pattern.select0(3), 5, "select0(3) - Alternating")
        with self.assertRaises(ValueError, msg="select10(4) should raise ValueError for '101010' bitvector"):
            bv_pattern.select10(4)
        print("Complex pattern operations passed.\n")

    def test_single_one(self):
        """Test Case 6: Single '1'"""
        print("Test Case 6: Single '1'")
        bv_single_one = BitVector("1")
        self.assertEqual(bv_single_one.rank1(0), 1, "rank1(0) - Single '1'")
        self.assertEqual(bv_single_one.rank10(0), 0, "rank10(0) - Single '1'")
        self.assertEqual(bv_single_one.select1(1), 0, "select1(1) - Single '1'")
        self.assertEqual(bv_single_one.rank0(0), 0, "rank0(0) - Single '1'")
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for single '1' bitvector"):
            bv_single_one.select10(1)
        with self.assertRaises(ValueError, msg="select0(1) should raise ValueError for single '1' bitvector"):
            bv_single_one.select0(1)
        print("Single '1' bitvector operations passed.\n")

    def test_single_zero(self):
        """Test Case 7: Single '0'"""
        print("Test Case 7: Single '0'")
        bv_single_zero = BitVector("0")
        self.assertEqual(bv_single_zero.rank1(0), 0, "rank1(0) - Single '0'")
        self.assertEqual(bv_single_zero.rank10(0), 0, "rank10(0) - Single '0'")
        self.assertEqual(bv_single_zero.rank0(0), 1, "rank0(0) - Single '0'")
        with self.assertRaises(ValueError, msg="select1(1) should raise ValueError for single '0' bitvector"):
            bv_single_zero.select1(1)
        with self.assertRaises(ValueError, msg="select10(1) should raise ValueError for single '0' bitvector"):
            bv_single_zero.select10(1)
        # For a single zero, select0 should return 0.
        self.assertEqual(bv_single_zero.select0(1), 0, "select0(1) - Single '0'")
        print("Single '0' bitvector operations passed.\n")

    def test_out_of_bounds_rank(self):
        """Test Case 8: Out-of-Bounds Indices for `rank`"""
        print("Test Case 8: Out-of-Bounds Indices for `rank`")
        bv = BitVector("1100101")
        self.assertEqual(bv.rank1(10), bv.total_ones, "rank1(10) - Out of Bounds")
        self.assertEqual(bv.rank10(10), bv.total_ten_patterns, "rank10(10) - Out of Bounds")
        print("Out-of-bounds rank operations passed.\n")

    def test_invalid_select(self):
        """Test Case 9: Invalid Occurrences for `select`"""
        print("Test Case 9: Invalid Occurrences for `select`")
        bv = BitVector("1100101")
        with self.assertRaises(ValueError, msg="select1(0) should raise ValueError"):
            bv.select1(0)
        with self.assertRaises(ValueError, msg="select1(5) should raise ValueError"):
            bv.select1(5)
        with self.assertRaises(ValueError, msg="select10(0) should raise ValueError"):
            bv.select10(0)
        with self.assertRaises(ValueError, msg="select10(3) should raise ValueError"):
            bv.select10(3)
        with self.assertRaises(ValueError, msg="select0(0) should raise ValueError"):
            bv.select0(0)
        print("Invalid select operations passed.\n")

    def test_large_bitvector_stress(self):
        """Test Case 10: Large BitVector Stress Test"""
        print("Test Case 10: Large BitVector Stress Test")
        large_bv = BitVector("101" * 1000)  # 3000-bit pattern
        self.assertEqual(large_bv.rank1(2999), 2000, "rank1(2999) - Large BitVector")
        self.assertEqual(large_bv.rank10(2999), 1000, "rank10(2999) - Large BitVector")
        self.assertEqual(large_bv.select1(500), 749, "select1(500) - Large BitVector")
        self.assertEqual(large_bv.select10(500), 1497, "select10(500) - Large BitVector")
        expected_rank0 = 3000 - 2000  # Total zeros in the 3000-bit pattern.
        self.assertEqual(large_bv.rank0(2999), expected_rank0, "rank0(2999) - Large BitVector")
        print("Large BitVector operations passed.\n")

if __name__ == '__main__':
    unittest.main()
