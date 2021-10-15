/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CK_GRIDWISE_GENERIC_2D_REDUCTION_MULTIBLOCK_HPP
#define CK_GRIDWISE_GENERIC_2D_REDUCTION_MULTIBLOCK_HPP

#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_threadwise.hpp"
#include "reduction_functions_blockwise.hpp"

#include "blockwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t BlockSize,
          typename srcDataType,
          typename dstDataType, // not used together with the beta input
          typename compType,
          typename src2dDescType,
          typename dst1dDescType,
          ReduceTensorOp_t op,
          NanPropagation_t nanPropaOpt,
          ReduceTensorIndices_t reduceIndicesOpt,
          index_t GredAccessesPerThreadInBlock>
struct GridwiseReduction_xy_to_x_multiblock
{
    static constexpr index_t inVectorSize =
        math::gcd(GredAccessesPerThreadInBlock, CK_PARAM_IN_VECTOR_IO_SIZE);

    using opReduce       = typename reduce_binary_operator<compType, op>::opType;
    using preUnaryOpType = typename reduce_unary_operator<compType, op, true, false>::preUnaryOp;
    using posUnaryOpType = typename reduce_unary_operator<compType, op, true, false>::posUnaryOp;

    static constexpr auto buffer1dDesc_1 =
        make_naive_tensor_descriptor_packed(make_tuple(Number<BlockSize>{}));

    using blockwise_reduce_1 = BlockwiseReduction_1d_block_buffer<decltype(buffer1dDesc_1),
                                                                  BlockSize,
                                                                  opReduce,
                                                                  nanPropaOpt>;

    static constexpr auto buffer1dDesc_2 = make_naive_tensor_descriptor_packed(
        make_tuple(Number<GredAccessesPerThreadInBlock * BlockSize>{}));

    using blockwise_reduce_2 = BlockwiseReduction_1d_block_buffer<decltype(buffer1dDesc_2),
                                                                  BlockSize,
                                                                  opReduce,
                                                                  nanPropaOpt>;

    static constexpr auto I0 = Number<0>{};

    static constexpr index_t BlockChunkSize = GredAccessesPerThreadInBlock * BlockSize;

    template <int RunId>
    __device__ static void Run(const src2dDescType& src2dDesc,
                               const dst1dDescType& dst1dDesc,
                               int origReduceLen,
                               int BlkGroupSize,
                               srcDataType alpha,
                               const srcDataType* const __restrict__ p_src_global,
                               dstDataType beta,
                               srcDataType* const __restrict__ ws_values_global,
                               int* const __restrict__ ws_indices_global);

    template <>
    __device__ static void Run<1>(const src2dDescType& src2dDesc,
                                  const dst1dDescType& dst1dDesc,
                                  int origReduceLen,
                                  int BlkGroupSize,
                                  srcDataType alpha,
                                  const srcDataType* const __restrict__ p_src_global,
                                  dstDataType beta,
                                  srcDataType* const __restrict__ ws_values_global,
                                  int* const __restrict__ ws_indices_global)
    {
        (void)ws_indices_global;

        (void)alpha; // unused
        (void)beta;  // unused

        const auto zeroVal = opReduce::GetReductionZeroVal();

        // LDS
        __shared__ compType p_block_reduce_buffer[BlockSize];

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>{}(zeroVal));
        auto workspace_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            ws_values_global, dst1dDesc.GetLength(I0) * BlkGroupSize);

        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, 1, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, GredAccessesPerThreadInBlock, true>
            in_thread_buf;

        using threadwise_reduce = ThreadReduce<decltype(in_thread_buf), opReduce, nanPropaOpt>;

        auto block_reduce_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_block_reduce_buffer, BlockSize);

        accuValue_buf(I0) = zeroVal;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider         = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / BlkGroupSize;
        const index_t block_local_id  = block_global_id % BlkGroupSize;

        const index_t reduceSizePerBlock =
            (((toReduceLength + BlkGroupSize - 1) / BlkGroupSize + BlockChunkSize - 1) /
             BlockChunkSize) *
            BlockChunkSize;

        using ThreadBufferLengths       = Sequence<1, GredAccessesPerThreadInBlock>;
        constexpr auto ThreadBufferDesc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<1>{}, Number<GredAccessesPerThreadInBlock>{}));

        auto threadwise_src_load = ThreadwiseTensorSliceTransfer_v2<srcDataType,
                                                                    compType,
                                                                    src2dDescType,
                                                                    decltype(ThreadBufferDesc),
                                                                    ThreadBufferLengths,
                                                                    Sequence<0, 1>,
                                                                    1,
                                                                    inVectorSize,
                                                                    1,
                                                                    false>(
            src2dDesc,
            make_multi_index(blkgroup_id,
                             block_local_id * reduceSizePerBlock +
                                 thread_local_id * GredAccessesPerThreadInBlock));

        constexpr auto in_thread_copy_step = make_multi_index(0, BlockChunkSize);

        const index_t toReduceChunks = reduceSizePerBlock / BlockChunkSize;

        for(index_t reducedChunks = 0; reducedChunks < toReduceChunks; reducedChunks++)
        {
            threadwise_src_load.Run(
                src2dDesc, src_global_buf, ThreadBufferDesc, make_tuple(I0, I0), in_thread_buf);

            // do element-wise pre-reduction operation
            threadwise_reduce::operate_on_elements(preUnaryOp, in_thread_buf);

            // do the reduction on the Thread Buffer
            threadwise_reduce::Reduce(in_thread_buf, accuValue_buf(I0));

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, in_thread_copy_step);
        }

        block_reduce_buf(thread_local_id) = accuValue_buf[I0];

        accuValue_buf(I0) = zeroVal;

        __syncthreads();

        blockwise_reduce_1::Reduce(block_reduce_buf, accuValue_buf(I0));

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

        const auto workspace_desc =
            make_naive_tensor_descriptor_packed(make_tuple(dst1dDesc.GetLength(I0) * BlkGroupSize));

        // The first thread in the block stores the reduced result to the global location
        // representing the block
        if(thread_local_id == 0)
        {
            auto threadwise_workspace_store =
                ThreadwiseTensorSliceTransfer_v1r3<compType,
                                                   srcDataType,
                                                   decltype(ReducedDataDesc),
                                                   decltype(workspace_desc),
                                                   Sequence<1>,
                                                   Sequence<0>,
                                                   0,
                                                   1,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(workspace_desc,
                                                         make_multi_index(block_global_id));

            threadwise_workspace_store.Run(ReducedDataDesc,
                                           make_tuple(I0),
                                           accuValue_buf,
                                           workspace_desc,
                                           workspace_global_buf);
        }
    };

    template <>
    __device__ static void Run<2>(const src2dDescType& src2dDesc,
                                  const dst1dDescType& dst1dDesc,
                                  int origReduceLen,
                                  int BlkGroupSize,
                                  srcDataType alpha,
                                  const srcDataType* const __restrict__ p_src_global,
                                  dstDataType beta,
                                  srcDataType* const __restrict__ ws_values_global,
                                  int* const __restrict__ ws_indices_global)
    {
        (void)alpha; // unused
        (void)beta;  // unused

        const auto zeroVal = opReduce::GetReductionZeroVal();

        // LDS
        __shared__ compType p_in_block_values_buffer[BlockChunkSize];
        __shared__ int p_in_block_indices_buffer[BlockChunkSize];

        const auto src_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>{}(zeroVal));
        auto workspace_global_val_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            ws_values_global, dst1dDesc.GetLength(I0) * BlkGroupSize);
        auto workspace_global_idx_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            ws_indices_global, dst1dDesc.GetLength(I0) * BlkGroupSize);

        auto in_block_val_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_in_block_values_buffer, BlockChunkSize);
        auto in_block_idx_buf =
            make_dynamic_buffer<AddressSpaceEnum_t::Lds>(p_in_block_indices_buffer, BlockChunkSize);
        StaticBuffer<AddressSpaceEnum_t::Vgpr, compType, 1, true> accuValue_buf;
        StaticBuffer<AddressSpaceEnum_t::Vgpr, int, 1, true> accuIndex_buf;

        accuValue_buf(I0) = zeroVal;
        accuIndex_buf(I0) = 0;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider         = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / BlkGroupSize;
        const index_t block_local_id  = block_global_id % BlkGroupSize;

        const index_t reduceSizePerBlock =
            (((toReduceLength + BlkGroupSize - 1) / BlkGroupSize + BlockChunkSize - 1) /
             BlockChunkSize) *
            BlockChunkSize;

        constexpr auto in_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}, Number<BlockChunkSize>{}));

        using ThreadSliceLengths   = Sequence<1, GredAccessesPerThreadInBlock>;
        using ThreadClusterLengths = Sequence<1, BlockSize>;

        auto blockwise_src_load = BlockwiseTensorSliceTransfer_v4<BlockSize,
                                                                  InMemoryDataOperationEnum_t::Set,
                                                                  Sequence<1, BlockChunkSize>,
                                                                  ThreadSliceLengths,
                                                                  ThreadClusterLengths,
                                                                  Sequence<0, 1>,
                                                                  srcDataType,
                                                                  compType,
                                                                  src2dDescType,
                                                                  decltype(in_block_desc),
                                                                  Sequence<0, 1>,
                                                                  Sequence<0, 1>,
                                                                  1,
                                                                  1,
                                                                  inVectorSize,
                                                                  inVectorSize,
                                                                  1,
                                                                  1,
                                                                  false,
                                                                  true>(
            src2dDesc,
            make_multi_index(blkgroup_id, block_local_id * reduceSizePerBlock),
            in_block_desc,
            make_multi_index(0, 0));

        constexpr auto in_block_copy_step = make_multi_index(0, BlockChunkSize);

        const index_t toReduceChunks = reduceSizePerBlock / BlockChunkSize;

        int indexOffset = block_local_id * reduceSizePerBlock;

        for(index_t reducedChunks = 0; reducedChunks < toReduceChunks; reducedChunks++)
        {
            blockwise_reduce_2::init_buffer_indices(in_block_idx_buf, indexOffset);

            blockwise_src_load.RunRead(src2dDesc, src_global_buf);
            blockwise_src_load.RunWrite(in_block_desc, in_block_val_buf);

            __syncthreads();

            // unary operation before reducing, needed by AMAX; For MIN/MAX, nothing is actually
            // done here
            blockwise_reduce_2::operate_on_elements(preUnaryOp, in_block_val_buf);

            blockwise_reduce_2::Reduce2(
                in_block_val_buf, in_block_idx_buf, accuValue_buf(I0), accuIndex_buf(I0));

            indexOffset += BlockChunkSize;

            blockwise_src_load.MoveSrcSliceWindow(src2dDesc, in_block_copy_step);
        }

        constexpr auto ReducedDataDesc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

        const auto workspace_desc =
            make_naive_tensor_descriptor_packed(make_tuple(dst1dDesc.GetLength(I0) * BlkGroupSize));

        // The first thread in the block stores the reduced result to the global location
        // representing the block
        if(thread_local_id == 0)
        {
            auto threadwise_workspace_val_store =
                ThreadwiseTensorSliceTransfer_v1r3<compType,
                                                   srcDataType,
                                                   decltype(ReducedDataDesc),
                                                   decltype(workspace_desc),
                                                   Sequence<1>,
                                                   Sequence<0>,
                                                   0,
                                                   1,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(workspace_desc,
                                                         make_multi_index(block_global_id));

            auto threadwise_workspace_idx_store =
                ThreadwiseTensorSliceTransfer_v1r3<int,
                                                   int,
                                                   decltype(ReducedDataDesc),
                                                   decltype(workspace_desc),
                                                   Sequence<1>,
                                                   Sequence<0>,
                                                   0,
                                                   1,
                                                   InMemoryDataOperationEnum_t::Set,
                                                   1,
                                                   true>(workspace_desc,
                                                         make_multi_index(block_global_id));

            threadwise_workspace_val_store.Run(ReducedDataDesc,
                                               make_tuple(I0),
                                               accuValue_buf,
                                               workspace_desc,
                                               workspace_global_val_buf);
            threadwise_workspace_idx_store.Run(ReducedDataDesc,
                                               make_tuple(I0),
                                               accuIndex_buf,
                                               workspace_desc,
                                               workspace_global_idx_buf);
        }
    };
};

} // namespace ck
#endif
